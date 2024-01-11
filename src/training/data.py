import os
import logging
import numpy as np
from PIL import Image
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from clip.clip import tokenize
import pickle
from PIL import ImageDraw, ImageFont
import random
import statistics
from textblob import TextBlob
import nltk


class PklDataset(Dataset):
    def __init__(self, input_filename, clip_preprocess, is_train=True, args=None):
        logging.debug(f'Loading pkl data from {input_filename}.')
        with open(input_filename, 'rb') as data_file:
            pkl_data = pickle.load(data_file)
        random.seed(args.seed)
        np.random.seed(args.seed)
        if args.token_stats:
            self.tokens_stats(pkl_data)
        if args.word_stats:
            self.word_stats(pkl_data)
        self.is_train = is_train
        self.args = args
        if args.data_mode == 'default':
            self.data = self.fold_data_default(pkl_data, args.fold)
        elif args.data_mode == 'cm':
            self.data = self.fold_data_om(pkl_data, args.fold)
        else:
            raise NotImplementedError(f'{args.data_mode} is not an option for data_mode param, '
                                      f'implement your own fold def for new data')
        if self.args.all_page_texts:
            self.text_batch = args.text_batch_size
        else:
            self.text_batch = 5
        self.clip_preprocess = clip_preprocess
        self.root_dir = os.path.dirname(input_filename)
        self.texts_count, self.figures_count, self.doc_count = None, None, None
        if self.is_train:
            self.count_elements()
            self.ind_to_figure = None
            self.prep_training_data()
        else:
            self.figure_index_doc, self.figure_index, self.text_index_doc, self.text_index, self.gt_matrix = [None] * 5
            self.count_elements()
            self.prep_val_data()
        logging.debug('Done loading data.')

    def aug_texts_demo(self, texts):
        phrases_list = self.special_phrases_list.copy()
        random.shuffle(phrases_list)
        aug_texts = []
        for text_dict in texts:
            random.shuffle(phrases_list)
            for phrase in phrases_list:
                if phrase.upper() in text_dict['text'].upper():
                    text_list = text_dict['text'].upper().split(' ')
                    phrase_idx = [i for i, sub_text in enumerate(text_list) if phrase.upper() in sub_text][0]
                    if phrase_idx == 1 or phrase_idx == 0:
                        start_idx = 0
                    else:
                        start_idx = max(np.random.randint(max(1, phrase_idx - 3), high=phrase_idx+1), 1)
                    high = min(phrase_idx + 4, len(text_list))
                    end_idx = min(np.random.randint(phrase_idx, high=high), len(text_list) - 1)
                    aug_text = ' '.join(text_list[start_idx:(end_idx+1)])
                    aug_text_dict = text_dict.copy()
                    if '' in aug_text:
                        continue
                    aug_text_dict['text'] = aug_text
                    aug_texts.append(aug_text_dict)
                    break
        return aug_texts

    def count_elements(self):
        figures_count = {}
        texts_count = {}
        doc_count = 0
        for page_dict in self.data.values():
            doc_number = page_dict['doc_ind']
            if doc_number not in figures_count.keys():
                figures_count[doc_number] = 0
                texts_count[doc_number] = 0
                doc_count += 1
            figures_count[doc_number] += len(page_dict['figures'])
            texts_count[doc_number] += len(page_dict['texts'])
        self.texts_count = texts_count
        self.figures_count = figures_count
        self.doc_count = doc_count

    def fold_data_om(self, pkl_data, fold):
        if fold == -1:
            return pkl_data
        all_docs_names = []
        for page_data in pkl_data.values():
            doc_name = page_data['figures'][0]['img_path'].split('/')[-2]
            if doc_name not in all_docs_names:
                all_docs_names.append(doc_name)
        if self.args.exp_mode == 'many':
            fold = 2
        fold_docs = {0: ['chevrolet'],  1: ['Mazda'], 2: ['nissan'], 3: ['renault'], 4: ['toyota']}
        fold_docs_names = [[doc for doc in all_docs_names if doc.startswith(doc_int)] for doc_int in fold_docs[fold]]
        fold_docs_names = [item for sublist in fold_docs_names for item in sublist]
        if self.args.exp_mode == 'many' or self.args.exp_mode == 'few':
            inner_fold_dict = {0: 1, 1: 4, 2: 2, 3: 1, 4: 3}
            inner_fold = inner_fold_dict[fold]
            start_ind = inner_fold * int(len(fold_docs_names) / 5)
            end_ind = min((inner_fold+1) * int(len(fold_docs_names) / 5), len(fold_docs_names))
            test_docs_names = fold_docs_names[start_ind:end_ind]
            train_docs_names = [doc_name for doc_name in fold_docs_names if doc_name not in test_docs_names]
        elif self.args.exp_mode == 'one':
            seed_fold_dict = {0: 2, 1: 1, 2: 2, 3: 4, 4: 0}
            random.seed(seed_fold_dict[fold])
            np.random.seed(seed_fold_dict[fold])
            test_docs_names = list(np.random.choice(fold_docs_names, len(fold_docs_names) - 1, replace=False))
            train_docs_names = [doc_name for doc_name in all_docs_names if doc_name not in test_docs_names]
        elif self.args.exp_mode == 'zero':
            test_docs_names = fold_docs_names
            train_docs_names = [doc_name for doc_name in all_docs_names if doc_name not in test_docs_names]
        else:
            raise NotImplementedError(f'{self.args.exp_mode} not supported for param exp_mode')
        self.test_docs_names = test_docs_names
        self.train_docs_names = train_docs_names
        test_docs_indices = [all_docs_names.index(doc_name) for doc_name in test_docs_names]
        train_docs_indices = [all_docs_names.index(doc_name) for doc_name in train_docs_names]
        data = {}
        print(f'Train docs names: {train_docs_names}')
        print(f'Test docs names: {test_docs_names}')
        if self.is_train:
            for (key, value) in pkl_data.items():
                if value['doc_ind'] in train_docs_indices:
                    assert value['figures'][0]['img_path'].split('/')[-2] == all_docs_names[value['doc_ind']]
                    data[key] = value
        else:
            page_num = 0
            for value in pkl_data.values():
                if value['doc_ind'] in test_docs_indices:
                    assert value['figures'][0]['img_path'].split('/')[-2] == all_docs_names[value['doc_ind']]
                    data[page_num] = value
                    page_num += 1
        return data

    def fold_data_default(self, pkl_data, fold):
        if fold == -1:  # For debugging purposes
            return pkl_data
        # doc_count = list(pkl_data.values())[-1]['doc_ind'] + 1
        fold = 2
        all_docs_names = []
        for page_data in pkl_data.values():
            doc_name = page_data['figures'][0]['img_path'].split('/')[-2]
            if doc_name not in all_docs_names:
                all_docs_names.append(doc_name)
        start_ind = fold * int(len(all_docs_names) / 5)
        end_ind = min((fold + 1) * int(len(all_docs_names) / 5), len(all_docs_names))
        test_docs_names = all_docs_names[start_ind:end_ind]
        train_docs_names = [doc_name for doc_name in all_docs_names if doc_name not in test_docs_names]
        self.test_docs_names = test_docs_names
        self.train_docs_names = train_docs_names
        test_docs_indices = [all_docs_names.index(doc_name) for doc_name in test_docs_names]
        train_docs_indices = [all_docs_names.index(doc_name) for doc_name in train_docs_names]
        data = {}
        print(f'Train docs names: {train_docs_names}')
        print(f'Test docs names: {test_docs_names}')
        if self.is_train:
            for (key, value) in pkl_data.items():
                if value['doc_ind'] in train_docs_indices:
                    assert value['figures'][0]['img_path'].split('/')[-2] == all_docs_names[value['doc_ind']]
                    data[key] = value
        else:
            page_num = 0
            for value in pkl_data.values():
                if value['doc_ind'] in test_docs_indices:
                    assert value['figures'][0]['img_path'].split('/')[-2] == all_docs_names[value['doc_ind']]
                    data[page_num] = value
                    page_num += 1
        return data

    def __len__(self):
        if self.is_train:
            if self.args.batch_mode == 'mix' or self.args.batch_mode == 'sep':
                return len(self.ind_to_figure)
            else:
                raise NotImplementedError(f'{self.args.batch_mode} isnt supported argument for batch_mode')
        else:
            return len(self.data)

    @classmethod
    def collate_fn_train(self):
        # noinspection PyUnreachableCode
        def fun(data):
            img, text_inputs = zip(*data)
            img = torch.stack(img, 0)
            text_inputs = torch.cat(text_inputs, 0)
            return img, text_inputs
        return fun

    @classmethod
    def collate_fn_val(self):
        # noinspection PyUnreachableCode
        def fun(data):
            img, text_inputs, image_labels, text_labels, img_paths, texts = zip(*data)
            img = torch.cat(img, 0)
            text_inputs = torch.cat(text_inputs, 0)
            image_labels = torch.cat(image_labels, 0)
            text_labels = torch.cat(text_labels, 0)
            img_paths = [path for path_list in img_paths for path in path_list]
            texts = [text for text_list in texts for text in text_list]
            return img, text_inputs, image_labels, text_labels, img_paths, texts
        return fun

    def __getitem__(self, idx):
        if self.is_train:
            if self.args.batch_mode == 'mix' or self.args.batch_mode == 'sep':
                images_batch, texts = self.get_item_train_mix(idx)
            else:
                raise NotImplementedError(f'{self.args.batch_mode} isnt supported argument for batch_mode')
            return images_batch, texts
        else:
            images_batch, text_inputs, image_labels, txt_labels, img_paths, texts = self.get_item_val(idx)
            return images_batch, text_inputs, image_labels, txt_labels, img_paths, texts

    @staticmethod
    def merge_boxes(boxes):
        # x0, y0, x1, y1; (left, bottom, right, top) ; 0,0 is bottom left
        if len(boxes) == 1:
            return boxes[0]
        boxes = np.stack(boxes)
        x_left = min(boxes[:, 0])
        y_bottom = min(boxes[:, 1])
        x_right = max(boxes[:, 2])
        y_above = max(boxes[:, 3])
        return [x_left, y_bottom, x_right, y_above]

    @staticmethod
    def merge_texts(boxes, texts):
        # x0, y0, x1, y1; (left, bottom, right, top) ; 0,0 is bottom left
        if len(texts) == 1:
            return texts[0]
        struct_array = np.array([(box[2], box[3]) for box in boxes], dtype=[("right", float), ("top", float)])
        sort_indices = np.argsort(struct_array, order=['top', 'right'])[::-1]
        return ' '.join([texts[i] for i in sort_indices])

    def merge_text_dicts(self, connected_comp_list, boxes, texts):
        new_texts_list = []
        for con_comp in connected_comp_list:
            con_boxes = [boxes[i] for i in con_comp]
            merged_box = self.merge_boxes(con_boxes)
            con_texts = [texts[i] for i in con_comp]
            merged_text = self.merge_texts(con_boxes, con_texts)
            new_texts_list.append({'bbox': merged_box, 'text': merged_text})
        return new_texts_list

    def get_item_train_mix(self, idx):
        page_ind, figure_ind = self.ind_to_figure[idx]
        page_dict = self.data[page_ind]
        figure_dict = page_dict['figures'][figure_ind]
        img_path = str(figure_dict['img_path'])
        if not os.path.exists(img_path):
            img_path = os.path.join(self.root_dir, str(figure_dict['img_path']))
        if not os.path.exists(img_path):
            img_path = os.path.join(self.root_dir, 'images', str(figure_dict['img_path']))
        # try:
        image = self.clip_preprocess(Image.open(img_path))
        # except Exception as err:
        #     print("preprocess transform error, opening image as RGB 256 BY 256")
        #     image = self.clip_preprocess(Image.new('RGB', (256, 256)))
        if not self.args.all_page_texts:
            chosen_texts_map = figure_dict['sides_text_map']
            if self.args.save_vis:
                self.save_vis(img_path, page_dict['figures'], figure_ind, page_dict['texts'], chosen_texts_map,
                              figure_dict['page_size'])
        else:
            chosen_texts_map = list(range(len(page_dict['texts'])))
            if self.args.save_vis:
                self.save_vis_no_sides(img_path, page_dict['figures'], figure_ind, page_dict['texts'],
                                       figure_dict['page_size'])
        if self.args.all_page_texts:
            texts = [texts_dict['text'] for texts_dict in page_dict['texts']]
            if len(texts) > self.text_batch:
                texts = texts[:self.text_batch]
        elif self.args.choose_one_baseline:
            texts = [texts_dict['text'] for texts_dict in page_dict['texts']]
            texts = list(np.random.choice(texts, 1))
        else:
            texts = [page_dict['texts'][i]['text'] for i in list(chosen_texts_map.values())]
        texts = tokenize(texts)
        if texts.shape[0] < self.text_batch and not self.args.choose_one_baseline:
            texts_num = texts.shape[0]
            for _ in range(self.text_batch - texts_num):
                pad = texts[np.random.randint(texts_num), :].unsqueeze(0)
                texts = torch.cat((texts, pad), dim=0)
        return image, texts

    def get_item_val(self, idx):
        page_dict = self.data[idx]
        images_batch = None
        images_paths = []
        for i, figure_dict in enumerate(page_dict['figures']):
            img_path = str(figure_dict['img_path'])
            if not os.path.exists(img_path):
                img_path = os.path.join(self.root_dir, str(figure_dict['img_path']))
            if not os.path.exists(img_path):
                img_path = os.path.join(self.root_dir, 'images', str(figure_dict['img_path']))
            # try:
            images = self.clip_preprocess(Image.open(img_path))
            # except Exception as err:
            #     print("preprocess transform error, opening image as RGB 256 BY 256")
            #     images = self.clip_preprocess(Image.new('RGB', (256, 256)))
            images_paths.append(img_path)
            if images_batch is None:
                images_batch = images.unsqueeze(0)
            else:
                images_batch = torch.cat((images_batch, images.unsqueeze(0)), dim=0)

            if not self.args.all_page_texts:
                chosen_texts_map = figure_dict['sides_text_map']
                self.gt_matrix[page_dict['doc_ind']][self.figure_index_doc[page_dict['doc_ind']],
                                                     [ind + self.text_index_doc[page_dict['doc_ind']] for ind in list(chosen_texts_map.values())]] = 1
                if self.args.save_vis:
                    self.save_vis(img_path, page_dict['figures'], i, page_dict['texts'], chosen_texts_map,
                                  figure_dict['page_size'])
            else:
                chosen_texts_map = list(range(len(page_dict['texts'])))
                self.gt_matrix[page_dict['doc_ind']][self.figure_index_doc[page_dict['doc_ind']],
                                                     [ind + self.text_index_doc[page_dict['doc_ind']] for ind in list(chosen_texts_map)]] = 1
                if self.args.save_vis:
                    self.save_vis_no_sides(img_path, page_dict['figures'], i, page_dict['texts'], figure_dict['page_size'])

            self.figure_index_doc[page_dict['doc_ind']] += 1
        texts = [text_dict['text'] for text_dict in page_dict['texts']]
        texts_emb = tokenize(texts)
        self.text_index_doc[page_dict['doc_ind']] += len(page_dict['texts'])
        image_labels = torch.ones(images_batch.shape[0], dtype=int) * page_dict['doc_ind']
        txt_labels = torch.ones(texts_emb.shape[0], dtype=int) * page_dict['doc_ind']
        return images_batch, texts_emb, image_labels, txt_labels, images_paths, texts

    def choose_side_texts(self, img_data, text_data, page_size):
        img_bbox = img_data['bbox']
        used_boxes = {}
        index_to_add = self.choose_text_by_side(img_bbox, text_data, page_size, used_boxes, side="left")
        if index_to_add is not None:
            used_boxes['left'] = index_to_add
        index_to_add = self.choose_text_by_side(img_bbox, text_data, page_size, used_boxes, side="above")
        if index_to_add is not None:
            used_boxes['above'] = index_to_add
        index_to_add = self.choose_text_by_side(img_bbox, text_data, page_size, used_boxes, side="below")
        if index_to_add is not None:
            used_boxes['below'] = index_to_add
        index_to_add = self.choose_text_by_side(img_bbox, text_data, page_size, used_boxes, side="right")
        if index_to_add is not None:
            used_boxes['right'] = index_to_add
        index_to_add = self.choose_text_by_side(img_bbox, text_data, page_size, used_boxes, side="overlap")
        if index_to_add is not None:
            used_boxes['overlap'] = index_to_add
        # index_to_add = self.choose_closest_text_by_side(img_bbox, text_data, used_boxes, side="left")
        # texts_data = [data_dict for i, data_dict in enumerate(data['text']) if i in used_boxes]
        # texts_data = [text_data[i] for i in used_boxes]
        return used_boxes

    @staticmethod
    def is_intersecting(box1, box2):
        x_left = max(box1[0], box2[0])
        y_bottom = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_above = min(box1[3], box2[3])
        if x_right < x_left or y_above < y_bottom:
            return 0
        else:
            return 1

    def choose_text_by_side(self, img_box, txt_data, page_size, used_boxes, side='right'):
        # x0, y0, x1, y1; (left, bottom, right, top) ; 0,0 is bottom left
        side_to_index_img = {'right': 2, 'left': 0, 'below': 1, 'above': 3, 'overlap': 3}
        side_to_index_text = {'right': 0, 'left': 2, 'below': 3, 'above': 1, 'overlap': 3}
        side_to_min_overlap = {'right': 0.05, 'left': 0.05, 'below': 0.05, 'above': 0.05, 'overlap': 0}
        min_overlap = side_to_min_overlap[side]
        valid = False
        min_dist = 11111
        min_dist_ind = 0
        for i, txt_dict in enumerate(txt_data):
            if i in used_boxes:
                continue
            # ignore subtitles and titles?
            # if txt_dict['type'] != 'paragraph':
            #     continue
            txt_box = txt_dict['bbox']
            # check that the bbox is in the correct side
            if side == 'above' or side == 'right':
                if txt_box[side_to_index_text[side]] < img_box[side_to_index_img[side]]:
                    continue
            elif side == 'below' or side == 'left':
                if txt_box[side_to_index_text[side]] > img_box[side_to_index_img[side]]:
                    continue
            else:  # side == overlap
                if not self.is_intersecting(txt_box, img_box):
                    continue
            # overlap is always between 0 to 1
            normalized_overlap = self.calc_overlap_2d(img_box, txt_box, page_size, side=side)
            if not normalized_overlap > min_overlap:
                continue
            if side == 'overlap':
                dist = normalized_overlap
            else:
                dist = self.calc_1d_dist(img_box, txt_box, page_size, side=side)
            if dist < min_dist:
                valid = True
                min_dist = dist
                min_dist_ind = i
        if not valid:
            return None
        return min_dist_ind

    @staticmethod
    def calc_1d_dist(img_box, txt_box, page_size, side="above"):
        # x0, y0, x1, y1; (left, bottom, right, top) ; 0,0 is bottom left
        side_to_img_index = {'right': 2, 'left': 0, 'below': 1, 'above': 3}
        side_to_txt_index = {'right': 2, 'left': 2, 'below': 3, 'above': 3}
        img_index = side_to_img_index[side]
        txt_index = side_to_txt_index[side]
        dist = min(abs(txt_box[txt_index] - img_box[img_index]), abs(txt_box[txt_index-2] - img_box[img_index]))
        if side == "above" or side == "below":
            max_dist = float(page_size[1])
        else:
            max_dist = float(page_size[0])
        return dist / max_dist

    @staticmethod
    def calc_overlap_2d(img_box, txt_box, page_size, side="above"):  # used right and left as example for above
        # x0, y0, x1, y1; (left, bottom, right, top) ; 0,0 is bottom left
        side_box = None
        if side == "above":
            side_box = [img_box[0], img_box[3], img_box[2], page_size[1]]
        elif side == "below":
            side_box = [img_box[0], 0, img_box[2], img_box[1]]
        elif side == "right":
            side_box = [img_box[2], img_box[1], page_size[0], img_box[3]]
        elif side == "left":
            side_box = [0, img_box[1], img_box[0], img_box[3]]
        elif side == "overlap":
            side_box = img_box
        x_left = max(side_box[0], txt_box[0])
        y_bottom = max(side_box[1], txt_box[1])
        x_right = min(side_box[2], txt_box[2])
        y_above = min(side_box[3], txt_box[3])
        if x_right < x_left or y_above < y_bottom:
            return 0.0
        intersection_area = (x_right - x_left) * (y_above - y_bottom)
        txt_box_area = (txt_box[2] - txt_box[0]) * (txt_box[3] - txt_box[1])
        return float(intersection_area) / float(txt_box_area)

    @staticmethod
    def calc_overlap_1d(img_box, txt_box, side="above"):  # used right and left as example for above
        # x0, y0, x1, y1; (left, bottom, right, top) ; 0,0 is bottom left
        if side == "above" or side == "below":
            overlap = max(0, min(img_box[2], txt_box[2]) - max(img_box[0], txt_box[0]))
            max_overlap = img_box[3] - img_box[1]

        else:
            overlap = max(0, min(img_box[3], txt_box[3]) - max(img_box[1], txt_box[1]))
            max_overlap = img_box[2] - img_box[0]
        return overlap / max_overlap

    # @staticmethod
    # def calc_overlap_alt(img_box, txt_box, side="above"):  # used right and left as example for above
    #     # x0, y0, x1, y1; (left, bottom, right, top) ; 0,0 is bottom left
    #     side_to_right_index = {'right': 3, 'left': 3, 'below': 2, 'above': 2}
    #     side_to_left_index = {'right': 1, 'left': 1, 'below': 0, 'above': 0}
    #     if side == "above" or side == "below":
    #         height = txt_box[3] - txt_box[1]
    #     else:
    #         height = txt_box[2] - txt_box[0]
    #     right = side_to_right_index[side]
    #     left = side_to_left_index[side]
    #     if (img_box[left] < txt_box[left]) and (img_box[right] > txt_box[left]):  # cases 2 and 4
    #         if img_box[right] < txt_box[right]:  # case 2
    #             overlap = img_box[right] - txt_box[left]
    #         else:  # case 4
    #             overlap = txt_box[right] - txt_box[left]
    #     elif (img_box[left] > txt_box[left]) and (img_box[left] < img_box[right]):  # cases 3 and 5
    #         if img_box[right] < txt_box[right]:  # case 3
    #             overlap = img_box[right] - img_box[left]
    #         else:  # case 5
    #             overlap = txt_box[right] - img_box[left]
    #     else:  # case 1 and 6
    #         # no horizontal overlap between boxes
    #         overlap = 0
    #     # return (overlap * height) / (box2[right] - box2[left])
    #     return overlap * height

    @staticmethod
    def choose_closest_text_by_side(img_box, txt_data, used_boxes, side='right'):
        # x0, y0, x1, y1; (left, bottom, right, top) ; 0,0 is bottom left
        side_to_index = {'right': 2, 'left': 0, 'below': 1, 'above': 3}
        side_to_index_text = {'right': 0, 'left': 2, 'below': 3, 'above': 1}
        min_dist = 100000
        min_dist_side = 100000
        min_dist_ind = 0
        min_dist_side_ind = 0
        valid = False
        img_box_center = [(img_box[2]+img_box[0])/2, (img_box[3]+img_box[1])/2]
        for i, txt_dict in enumerate(txt_data):
            if i in used_boxes:
                continue
            txt_box = txt_dict['bbox']
            txt_box_center = [(txt_box[2]+txt_box[0])/2, (txt_box[3]+txt_box[1])/2]
            dist = pow(img_box_center[0] - txt_box_center[0], 2) + pow(img_box_center[1] - txt_box_center[1], 2)
            if dist < min_dist:
                min_dist = dist
                min_dist_ind = i
            if txt_dict['type'] != 'paragraph':
                continue
            if dist >= min_dist_side:
                continue
            if side == 'right' or side == 'above':
                if img_box[side_to_index[side]] < txt_box[side_to_index_text[side]]:
                    valid = True
                    min_dist_side = dist
                    min_dist_side_ind = i
            elif side == 'left' or side == 'below':
                if img_box[side_to_index[side]] > txt_box[side_to_index_text[side]]:
                    valid = True
                    min_dist_side = dist
                    min_dist_side_ind = i
        if valid:
            return min_dist_side_ind
        else:
            return min_dist_ind

    def prep_training_data(self):
        # if self.args.batch_mode == 'mix':
        ind_to_figure = []
        for page_num, page_dict in self.data.items():
            for figure_num, figure_dict in enumerate(page_dict['figures']):
                ind_to_figure.append((page_num, figure_num))
        assert len(ind_to_figure) == sum(self.figures_count.values())
        self.ind_to_figure = ind_to_figure
        # elif self.args.batch_mode == 'sep':
        #     ind_to_figure = {}
        #     for page_num, page_dict in self.data.items():
        #         for figure_num, figure_dict in enumerate(page_dict['figures']):
        #             if page_dict['doc_ind'] not in ind_to_figure.keys():
        #                ind_to_figure[page_dict['doc_ind']] = []
        #             ind_to_figure[page_dict['doc_ind']].append((page_num, figure_num))
        #     self.ind_to_figure = ind_to_figure
        #     batch_indices = []
        #     for doc_ind, figures_list in ind_to_figure.items():
        #         start_ind = 0
        #         while start_ind < len(figures_list):
        #             batch_indices.append((doc_ind, start_ind, min(start_ind + self.args.batch_size, len(figures_list))))
        #             start_ind += self.args.batch_size
        #     self.batch_indices = batch_indices
        # else:
        #     raise NotImplementedError(f'{self.args.batch_mode} isnt a supported argument for batch_mode')

    def prep_val_data(self):
        self.figure_index_doc = {ind: 0 for ind in self.figures_count.keys()}
        self.figure_index = 0
        self.text_index_doc = {ind: 0 for ind in self.texts_count.keys()}
        self.text_index = 0
        self.gt_matrix = {ind: np.zeros((self.figures_count[ind], self.texts_count[ind]), dtype=int) for ind in self.figures_count.keys()}

    def save_vis(self, image_path, figures_data, figure_ind, texts_data, chosen_texts_indices, orig_dims):
        font = ImageFont.truetype(self.args.image_font_path, 40)
        pdf_file_path = '_'.join(image_path.replace('images', 'pages').split('_')[:-1]) + '.png'
        vis_save_path = image_path.replace('images', 'vis').split('.')[0] + '_vis.png'
        pdf_image = Image.open(pdf_file_path)
        pdf_image_for_print = pdf_image
        image_width_mult = float(pdf_image_for_print.size[0]) / float(orig_dims['width'])
        image_height_mult = float(pdf_image_for_print.size[1]) / float(orig_dims['height'])
        pdf_vis = ImageDraw.Draw(pdf_image_for_print)
        for i, figure_dict in enumerate(figures_data):
            figure_bbox = figure_dict['bbox']
            new_box = (np.floor(figure_bbox[0] * image_width_mult),
                       np.floor((float(orig_dims['height']) - figure_bbox[3]) * image_height_mult),
                       np.ceil(figure_bbox[2] * image_width_mult),
                       np.ceil((float(orig_dims['height']) - figure_bbox[1]) * image_height_mult))
            if i == figure_ind:
                color = "blue"
            else:
                color = "green"
            pdf_vis.rectangle(list(new_box), fill=None, outline=color, width=3)
        for i, text_dict in enumerate(texts_data):
            text_box = text_dict['bbox']
            box = (np.floor(text_box[0] * image_width_mult),
                   np.floor((float(orig_dims['height']) - text_box[3]) * image_height_mult),
                   np.ceil(text_box[2] * image_width_mult),
                   np.ceil((float(orig_dims['height']) - text_box[1]) * image_height_mult)
                   )
            if i in chosen_texts_indices.values():
                color = "red"
                side = list(chosen_texts_indices.keys())[list(chosen_texts_indices.values()).index(i)]
                pdf_vis.text((np.floor(text_box[0] * image_width_mult),
                              np.floor((float(orig_dims['height']) - text_box[3]) * image_height_mult)), side,
                             fill='orange', font=font)
            else:
                color = "orange"
            pdf_vis.rectangle(list(box), fill=None, outline=color, width=2)

        # pdf_image_for_print.save(vis_save_path)
        pdf_image_for_print.save(os.path.join(self.args.tmp_dir, os.path.basename(vis_save_path)))
        # shutil.copyfile(vis_save_path, os.path.join(self.args.tmp_dir, os.path.basename(vis_save_path)))
        del pdf_vis

    def save_vis_no_sides(self, image_path, figures_data, figure_ind, texts_data, orig_dims):
        font = ImageFont.truetype(self.args.image_font_path, 40)
        pdf_file_path = '_'.join(image_path.replace('images', 'pages').split('_')[:-1]) + '.png'
        vis_save_path = image_path.replace('images', 'vis').split('.')[0] + '_vis.png'
        pdf_image = Image.open(pdf_file_path)
        pdf_image_for_print = pdf_image
        image_width_mult = float(pdf_image_for_print.size[0]) / float(orig_dims['width'])
        image_height_mult = float(pdf_image_for_print.size[1]) / float(orig_dims['height'])
        pdf_vis = ImageDraw.Draw(pdf_image_for_print)
        for i, figure_dict in enumerate(figures_data):
            figure_bbox = figure_dict['bbox']
            new_box = (np.floor(figure_bbox[0] * image_width_mult),
                       np.floor((float(orig_dims['height']) - figure_bbox[3]) * image_height_mult),
                       np.ceil(figure_bbox[2] * image_width_mult),
                       np.ceil((float(orig_dims['height']) - figure_bbox[1]) * image_height_mult))
            if i == figure_ind:
                color = "blue"
            else:
                color = "green"
            pdf_vis.rectangle(list(new_box), fill=None, outline=color, width=3)
        for i, text_dict in enumerate(texts_data):
            text_box = text_dict['bbox']
            box = (np.floor(text_box[0] * image_width_mult),
                   np.floor((float(orig_dims['height']) - text_box[3]) * image_height_mult),
                   np.ceil(text_box[2] * image_width_mult),
                   np.ceil((float(orig_dims['height']) - text_box[1]) * image_height_mult)
                   )
            color = "orange"
            pdf_vis.rectangle(list(box), fill=None, outline=color, width=2)

        # pdf_image_for_print.save(vis_save_path)
        pdf_image_for_print.save(os.path.join(self.args.tmp_dir, os.path.basename(vis_save_path)))
        # shutil.copyfile(vis_save_path, os.path.join(self.args.tmp_dir, os.path.basename(vis_save_path)))
        del pdf_vis

    def gen_sim_val_gt(self, paths_per_doc):
        gt_matrix = self.gt_matrix
        # go over all docs
        for doc_ind, image_paths in paths_per_doc.items():
            meta_pkl = None
            img_name2ind = {os.path.basename(img_path): gt_line_ind for gt_line_ind, img_path in enumerate(image_paths)}
            # ind2img_name = {value: key for key, value in img_name2ind.items()}
            # go over all images in the doc
            for gt_line_ind, img_path in enumerate(image_paths):
                if meta_pkl is None:
                    doc_name = os.path.basename(os.path.dirname(img_path))
                    make_name = doc_name.split('_')[0].lower()
                    pkl_path = os.path.join(os.environ.get('MODELS_ROOT', 'pt_FETA_models'), 'cm_sim', make_name,
                                            doc_name, 'metadata.pkl')
                    if not os.path.exists(pkl_path):
                        print(f'{pkl_path} not found, skipping')
                        break
                    with open(pkl_path, 'rb') as data_file:
                        meta_pkl = pickle.load(data_file)
                img_name = os.path.basename(img_path)
                img_meta_ind = meta_pkl['img_name2ind'][img_name]
                page_ind, img_page_ind = meta_pkl['ind2sub_images'][img_meta_ind]
                img_dict = meta_pkl['images'][page_ind][img_page_ind]
                identical_images_indices = img_dict['identical']
                if identical_images_indices:
                    for id_img_ind in identical_images_indices:
                        id_page, id_page_ind = meta_pkl['ind2sub_images'][id_img_ind]
                        id_img_name = os.path.basename(meta_pkl['images'][id_page][id_page_ind]['img_path'])
                        id_text_gt_indices = np.where(gt_matrix[doc_ind][img_name2ind[id_img_name], :] == 1)[0]
                        gt_matrix[doc_ind][img_name2ind[img_name], id_text_gt_indices] = 1
        return gt_matrix

    @staticmethod
    def data_to_standard_format(data):
        text_list = []
        for data_entry in data.values():
            texts = [x['text'] for x in data_entry['texts']]
            for text in texts:
                text_list.append(text)
        return text_list

    def tokens_stats(self, data):
        text_list = self.data_to_standard_format(data)
        con_length = 1000
        tokens_num = []
        unique_tokens = torch.zeros(50000)
        for text in text_list:
            tok_text = tokenize(text, context_length=con_length)
            un_tokens = tok_text[0].unique()
            unique_tokens[un_tokens] = 1
            first_occ = torch.where(tok_text == 0)[1]
            if not first_occ.numel():
                tokens_num.append(con_length)
            else:
                tokens_num.append(int(first_occ[0]))
        mean = sum(tokens_num) / len(tokens_num)
        median = statistics.median(tokens_num)
        stddev = statistics.stdev(tokens_num)
        unique_tokens_num = unique_tokens.count_nonzero()
        print(f'Mean tokens per caption: {mean}, Median tokens per caption:{median}, stddev tokens per caption: {stddev}')
        print(f'unique number of tokens: {unique_tokens_num}')

    def word_stats(self, data):
        word_count = {}
        page_words_count_list = []
        figure_page_area_ratio_list = []
        for data_entry in data.values():
            page_words_count = 0
            texts = [x['text'] for x in data_entry['texts']]
            for text in texts:
                blob = TextBlob(text)
                page_words_count += len(blob.words)
                for word, count in blob.word_counts.items():
                    if word not in word_count.keys():
                        word_count[word] = 0
                    word_count[word] += count
            page_words_count_list.append(page_words_count)
            for figure_dict in data_entry['figures']:
                page_area = figure_dict['page_size']['height'] * figure_dict['page_size']['width']
                figure_area = (figure_dict['bbox'][2] - figure_dict['bbox'][0]) * (figure_dict['bbox'][3] - figure_dict['bbox'][1])
                figure_page_area_ratio = figure_area / float(page_area)
                figure_page_area_ratio_list.append(figure_page_area_ratio)
        total_word_count = sum(page_words_count_list)
        vocab_size = len(word_count.keys())
        word_vocab_ratio = float(total_word_count) / float(vocab_size)
        images_number = sum([len(data_entry['figures']) for data_entry in data.values()])
        texts_per_page_list = [len(data_entry['texts']) for data_entry in data.values()]
        average_captions_per_page = sum(texts_per_page_list) / len(texts_per_page_list)
        average_words_per_page = sum(page_words_count_list)/len(page_words_count_list)
        average_figure_page_area_ratio = sum(figure_page_area_ratio_list) / len(figure_page_area_ratio_list)
        word_count_sorted = [k for k, v in reversed(sorted(word_count.items(), key=lambda item: item[1]))]
        word_tags_sorted = [nltk.pos_tag(nltk.tokenize.word_tokenize(word)) for word in word_count_sorted]
        word_count_nouns = {word: float(count) for word, count in word_count.items() if nltk.pos_tag(nltk.tokenize.word_tokenize(word))[0][1]=='NN'}
        word_count_adj = {word: float(count) for word, count in word_count.items() if nltk.pos_tag(nltk.tokenize.word_tokenize(word))[0][1]=='JJ'}
        from wordcloud import WordCloud
        wc_nouns = WordCloud(background_color="white", max_words=20, relative_scaling=0.5,
                       normalize_plurals=False).generate_from_frequencies(word_count_nouns)
        wc_nouns.to_file(f'{self.args.tmp_dir}/wc_nouns.png')
        wc_adj = WordCloud(background_color="white", max_words=20, relative_scaling=0.5,
                       normalize_plurals=False).generate_from_frequencies(word_count_adj)
        wc_adj.to_file(f'{self.args.tmp_dir}/wc_adj.png')
        print(f'Average words per page: {average_words_per_page}')
        print(f'word count to vocabulary ration: {word_vocab_ratio}')
        print(f'Images number: {images_number}')
        print(f'Average captions per page: {average_captions_per_page}')
        print(f'average figure page area ratio: {average_figure_page_area_ratio}')


class PklDataset1Doc(PklDataset):
    def __init__(self, pkl_data, clip_preprocess, is_train=True, args=None, doc_num=0, root_dir=None):
        # logging.debug(f'Loading pkl data from {input_filename}.')
        # with open(input_filename, 'rb') as data_file:
        #     pkl_data = pickle.load(data_file)
        # df = pd.read_csv(input_filename, sep=sep)
        self.doc_num = doc_num
        self.is_train = is_train
        self.args = args
        self.special_phrases_list = ['cabinet', 'drawer', 'wall', 'floor', 'ceiling', 'glass', 'door', 'shelf',
                                     'shelving', 'bookcase', 'rack', 'wardrobe', 'chair', 'bed', 'couch', 'frame',
                                     'bin', 'stool', 'table', 'bench', 'box', 'remote', 'toy', 'kids', 'ledge',
                                     'duvet', 'curtain']
        self.data = self.doc_num_data(pkl_data, doc_num)
        if self.args.all_page_texts:
            self.text_batch = args.text_batch_size
        else:
            self.text_batch = 5
        # self.data = pkl_data
        # self.images = df[img_key].tolist()
        # self.captions = df[caption_key].tolist()
        self.clip_preprocess = clip_preprocess
        self.root_dir = root_dir
        random.seed(args.seed)
        np.random.seed(args.seed)
        # self.doc_count = list(self.data.values())[-1]['doc_ind']
        # figures_count = {ind: 0 for ind in range(self.doc_count + 1)}
        # texts_count = {ind: 0 for ind in range(self.doc_count + 1)}
        self.texts_count, self.figures_count, self.doc_count = None, None, None
        if self.is_train:
            self.count_elements()
            self.ind_to_figure = None
            self.prep_training_data()
        else:
            self.figure_index_doc, self.figure_index, self.text_index_doc, self.text_index, self.gt_matrix = [None] * 5
            self.count_elements()
            self.prep_val_data()
        logging.debug('Done loading data.')

    @staticmethod
    def doc_num_data(pkl_data, use_doc_num):
        new_data = {}
        for (key, value) in pkl_data.items():
            doc_number = value['doc_ind']
            if doc_number == use_doc_num:
                new_data[key] = value
        return new_data


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler


def get_pkl_dataset(args, preprocess_fn, is_train):
    if args.batch_mode == 'sep' and is_train:
        data_info = get_pkl_dataset_sep(args, preprocess_fn, is_train)
    elif args.batch_mode == 'mix' or args.batch_mode == 'sep':
        data_info = get_pkl_dataset_mix(args, preprocess_fn, is_train)
    else:
        raise NotImplementedError(f'{args.batch_mode} is not a valid option for batch_mode param')
    return data_info


def get_pkl_dataset_mix(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = PklDataset(input_filename, preprocess_fn, is_train=is_train, args=args)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    if is_train:
        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn_train(),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=is_train,
        )
    else:
        col_val = dataset.collate_fn_val()
        dataloader = DataLoader(
            dataset,
            collate_fn=col_val,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
            sampler=sampler,
            drop_last=is_train,
        )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, sampler)


def get_pkl_dataset_sep(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    fold_dataset = PklDataset(input_filename, preprocess_fn, is_train=is_train, args=args)
    data_loaders = []
    samplers = []
    doc_nums = list(fold_dataset.figures_count.keys())
    for doc_num in doc_nums:
        dataset = PklDataset1Doc(fold_dataset.data, preprocess_fn, is_train=is_train, args=args, doc_num=doc_num,
                                 root_dir=os.path.dirname(input_filename))
        num_samples = len(dataset)
        sampler = DistributedSampler(dataset) if args.distributed and is_train else None
        shuffle = is_train and sampler is None

        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn_train(),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=is_train,
        )
        dataloader.num_samples = num_samples
        dataloader.num_batches = len(dataloader)
        data_loaders.append(dataloader)
        samplers.append(sampler)
    return DataInfo(dataloader=DataCoLoader(data_loaders, args), sampler=samplers)


class DataCoLoaderIterator:
    def __init__(self, data_loaders, args):
        self.args = args
        self.data_loaders = data_loaders
        self.data_loaders_iters = [iter(d) for d in data_loaders]
        batches_indices = []
        for i, dataloader in enumerate(self.data_loaders):
            for j in range(dataloader.num_batches):
                batches_indices.append(i)
        self.batches_indices = batches_indices
        self.nLoaders = len(self.data_loaders_iters)
        self.done = np.zeros((self.nLoaders,), dtype=bool)

    def __next__(self):
        if self.batches_indices:
            ret = next(self.data_loaders_iters[self.batches_indices.pop(0)])
        else:
            raise StopIteration()
        return ret


class DataCoLoader:
    def __init__(self, data_loaders, args):
        self.args = args
        self.data_loaders = data_loaders
        self.num_batches = np.sum([(i, dataloader.num_batches) for i, dataloader in enumerate(self.data_loaders)])
        self.num_samples = np.sum([dataloader.num_samples for dataloader in self.data_loaders])

    def __iter__(self):
        return DataCoLoaderIterator(self.data_loaders, self.args)

    def __len__(self):
        return self.num_batches


def get_dataset_fn(data_path):
    ext = data_path.split('.')[-1]
    if ext in ['pkl']:
        return get_pkl_dataset
    else:
        raise ValueError(
            f"Tried to figure out dataset type, but failed for extention {ext}.")


def get_data(args, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data)(
            args, preprocess_train, is_train=True)
    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data)(
            args, preprocess_val, is_train=False)
    return data
