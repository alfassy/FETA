import logging
import os
import json
import glob
import argparse
import numpy as np
import pydevd_pycharm
import pickle
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import itertools
import shutil
import re
import PyPDF2


def parse_arguments():
    argparser = argparse.ArgumentParser(description="Prepare the image-data set for training or prediction")
    argparser.add_argument('--source-dir', required=True, help="path to directory with CCS groundtruth")
    argparser.add_argument('--target-dir', required=False, default="../data/pdf_data",
                           help="path to directory to store output groundtruth")
    argparser.add_argument('--name', required=False, default="",
                           help="path to directory to store Tesseract groundtruth")
    argparser.add_argument('--tmp_dir', required=False, default="/dccstor/alfassy/tmp",
                           help="path to save visualizations")
    argparser.add_argument('--font_path', type=str, default="./src/ocr_utils/arial.ttf",
                           help="Pil Image font path")
    argparser.add_argument('--mode', type=str, default="default",
                           help='data extraction mode to use, options: default,car_manuals')
    argparser.add_argument("--debug", default=False, action="store_true",
                           help="If true, entering debug mode in Pycharm")
    args = argparser.parse_args()
    return args


class DS_to_pkl():
    def __init__(self, args):
        self.resolution = 300
        self.args = args
        # self.source_dir = source_dir
        # self.target_dir = target_dir
        # self.resolution = 300

    def get_figure_texts(self, json_data):
        text_dicts = []
        for figure_dict in json_data['figures']:
            if figure_dict['cells']['data']:
                data_list = figure_dict['cells']['data']
                page_text_list = [text_data_list[-1] for text_data_list in data_list if len(text_data_list[-1]) > 4]
                text_boxes = [text_data_list[:4] for text_data_list in data_list if len(text_data_list[-1]) > 4]
                if not text_boxes:
                    continue
                page_index = figure_dict['prov'][0]['page']
                page_dim = json_data['page-dimensions'][page_index-1]
                connected_comp_list = self.get_connected_components(text_boxes, (page_dim['width'], page_dim['height']),
                                                                    add_perc=0.001)
                merged_text_list = self.merge_text_dicts(connected_comp_list, text_boxes, page_text_list)
                for merged_text_dict in merged_text_list:
                    main_text_dict = {'name': 'text', 'type': 'paragraph', 'text':  merged_text_dict['text'],
                                      'prov': [{'bbox': merged_text_dict['bbox'], 'page': page_index}]}
                    text_dicts.append(main_text_dict)
        return text_dicts

    def generate_default(self, min_fig_conf=0.9):
        # min_fig_conf - minimal confidence of DeepSearch's object detector in the bbox of the image
        source_dir = self.args.source_dir
        target_dir = self.args.target_dir
        target_images_dir = os.path.join(self.args.target_dir, 'images')
        target_pages_dir = os.path.join(self.args.target_dir, 'pages')
        target_vis_dir = os.path.join(self.args.target_dir, 'vis')
        if not os.path.exists(target_images_dir):
            os.makedirs(target_images_dir)
        if not os.path.exists(target_pages_dir):
            os.makedirs(target_pages_dir)
        if not os.path.exists(target_vis_dir):
            os.makedirs(target_vis_dir)
        pdf_filenames = glob.glob(os.path.join(source_dir, "*.pdf"))
        pdf_filenames += glob.glob(os.path.join(source_dir, "*.PDF"))
        # out = open(os.path.join(os.path.dirname(pdf_filenames[0]), "output.csv"), "w")
        # out_text = open(os.path.join(os.path.dirname(pdf_filenames[0]), "text_output.csv"), "w")
        # out.write("filepath\ttext\n")
        # creates a list of all figures, each figure has a list of dictionaries all the different texts from the same page.
        # figures_list = []
        # texts_list = []
        per_page_dict = {}
        doc_index = -1
        doc_page_index = -1
        # pdf_filenames[0], pdf_filenames[2] = pdf_filenames[2], pdf_filenames[0]
        pages_num = []
        figures_num = []
        texts_num = []
        for pdf_filename in pdf_filenames:
            print("processing {} ...".format(pdf_filename))
            pdf_basename = os.path.basename(pdf_filename).split('.')[0]
            json_filename = os.path.join(os.path.dirname(pdf_filename), pdf_basename + '.json')
            images_crops_folder = os.path.join(target_images_dir, pdf_basename)
            pages_folder = os.path.join(target_pages_dir, pdf_basename)
            vis_folder = os.path.join(target_vis_dir, pdf_basename)
            if not os.path.exists(images_crops_folder):
                os.makedirs(images_crops_folder)
            if not os.path.exists(pages_folder):
                os.makedirs(pages_folder)
            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder)
            if not os.path.exists(json_filename):
                print(f'No json file: {json_filename} for pdf: {pdf_filename}, skipping.')
                continue
            pages_images = self.pdf_to_png(pdf_filename)
            if not pages_images:
                print(f'Could not convert pdf: {pdf_filename} to images, skipping.')
                continue
            json_data = json_txt(json_filename, min_fig_conf)
            if len(json_data.text_data) == 0:
                print(f'No text in pdf: {pdf_filename}, skipping.')
                continue
            if len(json_data.figures_data) == 0:
                print(f'No figures in pdf: {pdf_filename}, skipping.')
                continue
            doc_index += 1
            # skipping tables
            for page_number, page_text_dict in json_data.text_data.items():
                text_boxes = [text_dict['bbox'] for text_dict in page_text_dict]
                page_dim = json_data.get_page_dims(page_number)
                connected_comp_list = json_data.get_connected_components(text_boxes,
                                                                         (page_dim['width'], page_dim['height']),
                                                                         add_perc=0.01)
                texts = [text_dict['text'] for text_dict in page_text_dict]
                merged_text_list = json_data.merge_text_dicts(connected_comp_list, text_boxes, texts)
                total_text_length = sum([len(text_i['text']) for text_i in merged_text_list])
                if total_text_length < 20:
                    print(f'Not enough total text in pdf: {pdf_basename} page {page_number}, skipping page')
                    continue
                merged_text_list_short = [text_dict for text_dict in merged_text_list if len(text_dict['text']) > 7]
                if merged_text_list_short:
                    merged_text_list = merged_text_list_short
                else:
                    print(f'All texts are short in pdf: {pdf_basename} page {page_number}, skipping page')
                    continue
                # Extract page images
                figures_list = []
                images_in_page = 0
                if page_number not in json_data.figures_data.keys():
                    print(f'No figures in: {pdf_basename} page {page_number}, skipping page')
                    continue
                for figure_dict in json_data.figures_data[page_number]:
                    page_figure_bbox = figure_dict['bbox']
                    # compute transform from PDF coordinates to pixel coordinates
                    image_figure_path = os.path.join(images_crops_folder, f'{page_number}_{images_in_page}.png')
                    page_figure_path = os.path.join(pages_folder, f'{page_number}.png')
                    vis_path = os.path.join(vis_folder, f'{page_number}_{images_in_page}_vis.png')
                    image_figure_path_relative = os.path.join('images', pdf_basename,
                                                              f'{page_number}_{images_in_page}.png')
                    self.crop_and_save_image(pages_images[page_number-1], page_figure_bbox, page_dim, page_figure_path,
                                             image_figure_path, merged_text_list)
                    if images_in_page == 0:
                        doc_page_index += 1
                        per_page_dict[doc_page_index] = {"texts": None, "figures": [], "doc_ind": doc_index}

                    self.save_vis(page_figure_bbox, merged_text_list, page_dim, page_figure_path,
                                  vis_path)
                    figures_list.append({'img_path': image_figure_path_relative, 'bbox': page_figure_bbox,
                                         'page_size': page_dim, 'doc_page_key': doc_page_index})
                    images_in_page += 1
                if images_in_page > 0:
                    per_page_dict[doc_page_index]["figures"] = figures_list
                    per_page_dict[doc_page_index]["texts"] = merged_text_list
        print(f'docs_num :{len(pdf_filenames)}')
        output_filepath = os.path.join(target_dir, f'{self.args.name}.pkl')
        print(f'Saving per page dictionary to: {output_filepath}')
        with open(output_filepath, 'wb') as out_file:
            pickle.dump(per_page_dict, out_file, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    def generate_car_manuals(self, min_fig_conf=0.9):
        source_dir = self.args.source_dir
        target_dir = self.args.target_dir
        target_images_dir = os.path.join(self.args.target_dir, 'images')
        target_pages_dir = os.path.join(self.args.target_dir, 'pages')
        target_vis_dir = os.path.join(self.args.target_dir, 'vis')
        if not os.path.exists(target_images_dir):
            os.makedirs(target_images_dir)
        if not os.path.exists(target_pages_dir):
            os.makedirs(target_pages_dir)
        if not os.path.exists(target_vis_dir):
            os.makedirs(target_vis_dir)
        pdf_filenames = glob.glob(os.path.join(source_dir, "*.pdf"))
        pdf_filenames += glob.glob(os.path.join(source_dir, "*.PDF"))
        # out = open(os.path.join(os.path.dirname(pdf_filenames[0]), "output.csv"), "w")
        # out_text = open(os.path.join(os.path.dirname(pdf_filenames[0]), "text_output.csv"), "w")
        # out.write("filepath\ttext\n")
        # creates a list of all figures, each figure has a list of dictionaries all the different texts from the same page.
        # figures_list = []
        # texts_list = []
        per_page_dict = {}
        doc_index = -1
        doc_page_index = -1
        pages_num = []
        figures_num = []
        texts_num = []
        max_pages = 0
        for pdf_filename in pdf_filenames:
            print("processing {} ...".format(pdf_filename))
            pdf_basename = os.path.basename(pdf_filename).split('.')[0]
            json_filename = os.path.join(os.path.dirname(pdf_filename), pdf_basename + '.json')
            images_crops_folder = os.path.join(target_images_dir, pdf_basename)
            pages_folder = os.path.join(target_pages_dir, pdf_basename)
            vis_folder = os.path.join(target_vis_dir, pdf_basename)
            if not os.path.exists(images_crops_folder):
                os.makedirs(images_crops_folder)
            if not os.path.exists(pages_folder):
                os.makedirs(pages_folder)
            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder)
            if not os.path.exists(json_filename):
                print(f'No json file: {json_filename} for pdf: {pdf_filename}, skipping.')
                continue
            # root = os.path.basename(pdf_filename).replace(".ann", ".pdf")
            # pdf_filename = pdf_filename.replace(".ann", ".pdf")
            # raw_filename = pdf_filename.replace(".ann", ".raw")
            pages_images = self.pdf_to_png(pdf_filename)
            if not pages_images:
                print(f'Could not convert pdf: {pdf_filename} to images, skipping.')
                continue
            if len(pages_images) < 20:
                print(f'Pdf: {pdf_filename} is less than 20 pages, skipping.')
                continue
            with open(json_filename, "r") as fd:
                json_data = json.load(fd)
            if len(json_data['page-dimensions']) != json_data['page-dimensions'][-1]['page']:
                raise RuntimeError(f'pdf {pdf_filename} has page dims error')
            # if json_data['page-dimensions'][-1]['page'] < 20:
            #     print(f'Pdf: {pdf_filename} is less than 20 pages, skipping.')
            #     continue
            # skipping json lines with no usable text
            text_types_skip = ['table', 'figure', 'page-footer', 'page-header']
            figure_texts = self.get_figure_texts(json_data)
            main_text_count = 0
            main_text_index = 0
            for figure_dict in figure_texts:
                figure_page_num = figure_dict['prov'][0]['page']
                while (main_text_index < len(json_data['main-text'])) and ((json_data['main-text'][main_text_index]['type'] in text_types_skip) or \
                        (json_data['main-text'][main_text_index]['prov'][0]['page'] < figure_page_num)):
                    if json_data['main-text'][main_text_index]['type'] not in text_types_skip:
                        main_text_count += 1
                    main_text_index += 1
                json_data['main-text'].insert(main_text_index, figure_dict)


            doc_index += 1
            page_index = 1
            figures_index = 0
            main_text_index = 0
            while main_text_index < len(json_data['main-text']):
                if json_data['main-text'][main_text_index]['type'] not in text_types_skip:
                    json_data['main-text'][main_text_index]['text'] = \
                        re.sub('[^\u00C0-\u017FA-Za-z0-9"!@#$%^&*()\[\]{};:,./<>?\\\|`~\-=_+/s]+', ' ',
                               json_data['main-text'][main_text_index]['text'])
                main_text_index += 1
            main_text_index = 0
            while (main_text_index < len(json_data['main-text'])) and \
                    (json_data['main-text'][main_text_index]['type'] in text_types_skip):
                main_text_index += 1
            if main_text_index >= len(json_data['main-text']):
                print(f'No text in pdf {pdf_filename}')
                continue
            text_page = json_data['main-text'][main_text_index]['prov'][0]['page']
            # print("page: ", )
            if json_data['page-dimensions'][-1]['page'] > max_pages:
                max_pages = json_data['page-dimensions'][-1]['page']
            pages_num.append(json_data['page-dimensions'][-1]['page'])
            figures_num.append(len(json_data['figures']))
            texts_num.append(len(json_data['main-text']))
            # dims = page["dimensions"]
            # cells = page["cells"]

            # pw = float(dims["width"])
            # ph = float(dims["height"])

            # items=[]
            # for cell in page["cells"]:
            # for cell in page["cells"]:
            print(f'Skipping tables')
            # TODO: test if page_index fits page from json
            while (main_text_index < len(json_data['main-text'])) and (page_index <= len(json_data['page-dimensions'])):
                # Retrieve all texts in a page and put in a single list, dictionary for each separate text.
                page_text_list = []
                while (main_text_index < len(json_data['main-text'])) and (text_page <= page_index):
                    if text_page == page_index:
                        page_text_list.append({'text': json_data['main-text'][main_text_index]['text'],
                                               'bbox': json_data['main-text'][main_text_index]['prov'][0]['bbox'],
                                               'type': json_data['main-text'][main_text_index]['type']})
                    main_text_index += 1
                    # get next main_text_index which has usable text
                    while (main_text_index < len(json_data['main-text'])) and \
                            (json_data['main-text'][main_text_index]['type'] in text_types_skip):
                        main_text_index += 1
                    if main_text_index < len(json_data['main-text']):
                        text_page = json_data['main-text'][main_text_index]['prov'][0]['page']
                if not page_text_list:
                    print(f'No text in pdf: {pdf_basename} page {page_index}, skipping page')
                    page_index += 1
                    continue
                text_boxes = [text_dict['bbox'] for text_dict in page_text_list]
                page_dim = json_data['page-dimensions'][page_index-1]
                connected_comp_list = self.get_connected_components(text_boxes, (page_dim['width'], page_dim['height']),
                                                                    add_perc=0.01)
                texts = [text_dict['text'] for text_dict in page_text_list]
                merged_text_list = self.merge_text_dicts(connected_comp_list, text_boxes, texts)
                total_text_length = sum([len(text_i['text']) for text_i in merged_text_list])
                if total_text_length < 20:
                    print(f'Not enough total text in pdf: {pdf_basename} page {page_index}, skipping page')
                    page_index += 1
                    continue
                merged_text_list_short = [text_dict for text_dict in merged_text_list if len(re.sub('[^\u00C0-\u017FA-Za-z0-9"!@#$%^&*()\[\]{};:,./<>?\\\|`~\-=_+/s]+', '',
                                                                                                    text_dict['text'])) > 12]
                if merged_text_list_short:
                    merged_text_list = merged_text_list_short
                else:
                    print(f'All texts are short in pdf: {pdf_basename} page {page_index}, skipping page')
                    page_index += 1
                    continue
                used_indices = []
                figures_list = []
                chosen_texts_indices = {}
                images_in_page = 0
                while (figures_index < len(json_data['figures'])) and \
                        (json_data['figures'][figures_index]['prov'][0]['page'] <= page_index):
                    if json_data['figures'][figures_index]['type'] != 'picture':
                        raise NotImplementedError("Code only support figures of type picture")
                    if json_data['figures'][figures_index]['prov'][0]['page'] == page_index:
                        if json_data['figures'][figures_index]['confidence'] >= min_fig_conf:
                            if json_data['figures'][figures_index]['cells']['data']:
                                figures_index += 1
                                print("Skipping header in figures key")
                                continue
                            page_figure_bbox = json_data['figures'][figures_index]['prov'][0]['bbox']  # [left, bottom, right, top] bottom left is 0,0
                            # Validate CCS bbox, left < right and bottom < top
                            if page_figure_bbox[0] > page_figure_bbox[2] or page_figure_bbox[1] > page_figure_bbox[3]:
                                figures_index += 1
                                print(f'Bad bbox {page_figure_bbox} in pdf: {pdf_basename} page {page_index}, skipping')
                                continue
                            page_area = page_dim['height'] * page_dim['width']
                            figure_area = (page_figure_bbox[2] - page_figure_bbox[0]) * (page_figure_bbox[3] - page_figure_bbox[1])
                            figure_page_area_ratio = figure_area / page_area
                            if figure_page_area_ratio > 0.65:
                                figures_index += 1
                                print(f'figure is more than 65 percent of the page in pdf: {pdf_basename} page {page_index}, skipping')
                                continue
                            if figure_page_area_ratio < 0.05:
                                figures_index += 1
                                print(f'figure is less than 5 percent of the page in pdf: {pdf_basename} page {page_index}, skipping')
                                continue
                            chosen_texts = self.choose_side_texts(page_figure_bbox, merged_text_list,
                                                                  (page_dim['width'], page_dim['height']))
                            if len(chosen_texts) == 0:
                                figures_index += 1
                                print(f'No related text found in pdf: {pdf_basename} page {page_index},  bbox: {page_figure_bbox}, skipping')
                                continue
                            # compute transform from PDF coordinates to pixel coordinates
                            image_figure_path = os.path.join(images_crops_folder, f'{page_index}_{images_in_page}.png')
                            page_figure_path = os.path.join(pages_folder, f'{page_index}.png')
                            vis_path = os.path.join(vis_folder, f'{page_index}_{images_in_page}_vis.png')
                            image_figure_path_relative = os.path.join('images', pdf_basename,
                                                                      f'{page_index}_{images_in_page}.png')
                            if not os.path.exists(image_figure_path):
                                self.crop_and_save_image(pages_images[page_index-1], page_figure_bbox,
                                                         json_data['page-dimensions'][page_index-1], page_figure_path,
                                                         image_figure_path, merged_text_list)
                            if images_in_page == 0:
                                doc_page_index += 1
                                per_page_dict[doc_page_index] = {"texts": None, "figures": [], "doc_ind": doc_index}
                                # per_page_dict[doc_page_index]['texts'] = merged_text_list
                                # for text_dict in page_text_list:
                                #     per_page_dict[doc_page_index]["texts"].append(text_dict)

                            chosen_texts_indices[images_in_page] = chosen_texts
                            used_indices.append(chosen_texts.values())
                            # per_page_dict[doc_page_index]["figures"].append(
                            #     {'img_path': image_figure_path, 'bbox': page_figure_bbox,
                            #      'page_size': json_data['page-dimensions'][page_index-1],
                            #      'doc_page_key': doc_page_index})
                            self.save_vis(page_figure_bbox, merged_text_list,
                                          json_data['page-dimensions'][page_index-1], page_figure_path, vis_path)
                            figures_list.append({'img_path': image_figure_path_relative, 'bbox': page_figure_bbox,
                                                 'page_size': json_data['page-dimensions'][page_index-1],
                                                 'doc_page_key': doc_page_index})
                            # figures_list.append({'img_path': image_figure_path, 'bbox': page_figure_bbox,
                            #                      'page_size': json_data['page-dimensions'][page_index-1],
                            #                      'doc_page_key': doc_page_index, 'text': page_text_list})
                            # page_images_list.append({'img_path': image_figure_path, 'bbox': page_figure_bbox})
                            images_in_page += 1
                        else:
                            print(f'Image in pdf: {pdf_basename} page {page_index}, has confidence lower than {min_fig_conf}, skipping image')
                    figures_index += 1
                if images_in_page > 0:
                    used_texts_map = {}
                    chosen_texts_list = []
                    for figure_num, chosen_texts_dict in chosen_texts_indices.items():
                        new_chosen_texts_dict = {}
                        for side, old_text_ind in chosen_texts_dict.items():
                            if old_text_ind not in used_texts_map.keys():
                                chosen_texts_list.append(merged_text_list[old_text_ind])
                                used_texts_map[old_text_ind] = len(chosen_texts_list) - 1
                            new_chosen_texts_dict[side] = used_texts_map[old_text_ind]
                        figures_list[figure_num]["sides_text_map"] = new_chosen_texts_dict
                        per_page_dict[doc_page_index]["figures"].append(figures_list[figure_num])
                    per_page_dict[doc_page_index]["texts"] = chosen_texts_list

                page_index += 1
        avg_pages_num = sum(pages_num) / len(pages_num)
        avg_figures_num = sum(figures_num) / len(figures_num)
        avg_texts_num = sum(texts_num) / len(texts_num)
        print(f'Average number of pages: {avg_pages_num}')
        print(f'Average number of figures: {avg_figures_num}')
        print(f'Average number of texts: {avg_texts_num}')
        output_filepath = os.path.join(target_dir, f'{self.args.name}.pkl')
        print(f'Saving per page dictionary to: {output_filepath}')
        with open(output_filepath, 'wb') as out_file:
            pickle.dump(per_page_dict, out_file, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    def save_vis(self, figure_bbox, texts_data, orig_dims, page_path, vis_save_path):
        # text_side = ['left', 'above', 'below', 'right']
        # font = ImageFont.truetype(os.path.join(os.getcwd(), self.args.font_path), 40)
        # pdf_file_path = '_'.join(image_path.split('_')[:-1]) + '.png'
        # vis_save_path = image_path.split('.')[0] + '_vis.png'
        pdf_image = Image.open(page_path)
        pdf_image_for_print = pdf_image
        image_width_mult = float(pdf_image_for_print.size[0]) / float(orig_dims['width'])
        image_height_mult = float(pdf_image_for_print.size[1]) / float(orig_dims['height'])
        pdf_vis = ImageDraw.Draw(pdf_image_for_print)
        new_box = (np.floor(figure_bbox[0] * image_width_mult),
                   np.floor((float(orig_dims['height']) - figure_bbox[3]) * image_height_mult),
                   np.ceil(figure_bbox[2] * image_width_mult),
                   np.ceil((float(orig_dims['height']) - figure_bbox[1]) * image_height_mult))
        pdf_vis.rectangle(list(new_box), fill=None, outline="blue", width=3)
        for i, text_dict in enumerate(texts_data):
            text_box = text_dict['bbox']
            box = (np.floor(text_box[0] * image_width_mult),
                   np.floor((float(orig_dims['height']) - text_box[3]) * image_height_mult),
                   np.ceil(text_box[2] * image_width_mult),
                   np.ceil((float(orig_dims['height']) - text_box[1]) * image_height_mult)
                   )
            pdf_vis.rectangle(list(box), fill=None, outline="red", width=2)
        pdf_image_for_print.save(vis_save_path)
        # cv2.imwrite(vis_save_path.split('.')[0]+".png", cv2.imread(vis_save_path))
        # cv2.imwrite(os.path.join(self.args.tmp_dir, os.path.basename(vis_save_path)), cv2.imread(vis_save_path))
        shutil.copyfile(vis_save_path, os.path.join(self.args.tmp_dir, os.path.basename(vis_save_path)))
        del pdf_vis

    def crop_and_save_image(self, pdf_image, bbox, orig_dims, pdf_save_path, image_save_path, page_text_list):
        # bbox - [left, bottom, right, top] bottom left is 0,0.
        # crop image from pdf image, save separately, return path to image.
        pdf_image_for_print = pdf_image.copy()
        image_width_mult = float(pdf_image_for_print.size[0]) / float(orig_dims['width'])
        image_height_mult = float(pdf_image_for_print.size[1]) / float(orig_dims['height'])

        box = (np.floor(bbox[0] * image_width_mult),
               np.floor((float(orig_dims['height']) - bbox[3]) * image_height_mult),
               np.ceil(bbox[2] * image_width_mult),
               np.ceil((float(orig_dims['height']) - bbox[1]) * image_height_mult)
               )
        # This will crop out a single cell from the page image and save it. You can instead do other things with the cell `box` in pixel coordinates here.
        cropped_image = pdf_image_for_print.crop(box=box)  # Expects box [left, top, right, bottom], top left is 0,0
        # pdf_vis = ImageDraw.Draw(pdf_image_for_print)
        # pdf_vis.rectangle(list(box), fill=None, outline="blue", width=3)
        # for text_dict in page_text_list:
        #     text_box = text_dict['bbox']
        #     box = (np.floor(text_box[0] * image_width_mult),
        #            np.floor((float(orig_dims['height']) - text_box[3]) * image_height_mult),
        #            np.ceil(text_box[2] * image_width_mult),
        #            np.ceil((float(orig_dims['height']) - text_box[1]) * image_height_mult)
        #            )
        #     pdf_vis.rectangle(list(box), fill=None, outline="red", width=2)

        #cell_.show()
        # cnt += 1
        # gt_png = os.path.join(target_dir, "cell={0:0>7}.tif".format(cnt))
        # gt_txt = os.path.join(target_dir, "cell={0:0>7}.gt.txt".format(cnt))
        pdf_image_for_print.save(pdf_save_path)
        # cv2.imwrite(pdf_save_path.split('.')[0]+".png", cv2.imread(pdf_save_path))
        cropped_image.save(image_save_path)
        # cv2.imwrite(image_save_path.split('.')[0]+".png", cv2.imread(image_save_path))
        # del pdf_vis

    def pdf_to_png(self, pdf_filename):
        pages_images = convert_from_path(pdf_filename, fmt="png", use_cropbox=True)
        return pages_images

    def choose_side_texts(self, img_bbox, text_data, page_size):
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

    def merge_text_dicts(self, connected_comp_list, boxes, texts):
        new_texts_list = []
        for con_comp in connected_comp_list:
            con_boxes = [boxes[i] for i in con_comp]
            merged_box = self.merge_boxes(con_boxes)
            con_texts = [texts[i] for i in con_comp]
            merged_text = self.merge_texts(con_boxes, con_texts)
            new_texts_list.append({'bbox': merged_box, 'text': merged_text})
        return new_texts_list

    @staticmethod
    def merge_texts(boxes, texts):
        # x0, y0, x1, y1; (left, bottom, right, top) ; 0,0 is bottom left
        if len(texts) == 1:
            return texts[0]
        struct_array = np.array([(box[2], box[3]) for box in boxes], dtype=[("right", float), ("top", float)])
        sort_indices = np.argsort(struct_array, order=['top', 'right'])[::-1]
        return ' '.join([texts[i] for i in sort_indices])

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

    def get_connected_components(self, txt_boxes, page_size, add_perc=0.05):
        # x0, y0, x1, y1; (left, bottom, right, top) ; 0,0 is bottom left
        boxes = np.stack(txt_boxes)
        # boxes_percentile_vert = add_perc * (boxes[:, 3] - boxes[:, 1])
        boxes_percentile_vert = add_perc * page_size[1]
        boxes_percentile_horiz = boxes_percentile_vert / 4
        # boxes_percentile_horiz = add_perc * (boxes[:, 2] - boxes[:, 0])
        boxes[:, 0] = np.maximum(np.zeros_like(boxes[:, 0]), boxes[:, 0] - boxes_percentile_horiz)
        boxes[:, 2] = np.minimum(np.ones_like(boxes[:, 2]) * page_size[0], boxes[:, 2] + boxes_percentile_horiz)
        boxes[:, 1] = np.maximum(np.zeros_like(boxes[:, 1]), boxes[:, 1] - boxes_percentile_vert)
        boxes[:, 3] = np.minimum(np.ones_like(boxes[:, 3]) * page_size[1], boxes[:, 3] + boxes_percentile_vert)
        # get connected components if boxes are intersecting
        all_combs = list(itertools.combinations(np.arange(boxes.shape[0]), 2))
        g = Graph(boxes.shape[0])
        for i, j in all_combs:
            is_inter = self.is_intersecting(boxes[i, :], boxes[j, :])
            if is_inter:
                g.addEdge(i, j)
        cc, cc_dict = g.connectedComponents()
        return cc

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

    def split_pdf_by_bookmarks(self, path):
        del_to_identifier = {'Earth': 'MS', 'Life': 'Chapter', 'Physical': ' '}
        for keyword, delimiter in del_to_identifier.items():
            if keyword in path:
                start_del = delimiter
                break
        bookmarks = self.get_pdf_bookmarks(path)
        file_name = os.path.basename(path).split('.')[0].replace(' ', '_')
        pdf_chapters = {pg: file_name + '_' + bk.replace(' ', '_') for pg, bk in bookmarks.items() if
                        bk.startswith(start_del)}
        page_start_list = list(pdf_chapters.keys())
        pdf_chapters_list = list(pdf_chapters.values())
        reader = PyPDF2.PdfFileReader(open(path, "rb"))
        pdf_page_num = reader.numPages
        for i, chapter in enumerate(pdf_chapters_list):
            start_page = page_start_list[i]
            if i == len(pdf_chapters_list) - 1:
                end_page = pdf_page_num - 1
            else:
                end_page = page_start_list[i + 1]
            output = PyPDF2.PdfFileWriter()
            for page_i in range(start_page, end_page):
                output.addPage(reader.getPage(page_i))

            with open(os.path.join('data/ck12_chapters', f'{chapter}.pdf'), "wb") as outputStream:
                output.write(outputStream)

    @staticmethod
    def get_pdf_bookmarks(path):
        def bookmark_dict(bookmark_list):
            result = {}
            for item in bookmark_list:
                if isinstance(item, list):
                    continue
                    # recursive call
                    result.update(bookmark_dict(item))
                else:
                    result[reader.getDestinationPageNumber(item)] = item.title
            return result

        reader = PyPDF2.PdfFileReader(path)

        return bookmark_dict(reader.getOutlines())


class json_txt:
    def __init__(self, json_filename, min_fig_conf):
        with open(json_filename, "r") as fd:
            json_data = json.load(fd)
        self.json_data = json_data
        pdf_basename = os.path.basename(json_filename).split('.')[0]
        if len(json_data['page-dimensions']) != json_data['page-dimensions'][-1]['page']:
            raise RuntimeError(f'pdf {pdf_basename} has page dims error')
        page_texts_dict = {}
        # skipping json lines with no usable text
        self.text_types_skip = ['table', 'figure', 'page-footer', 'page-header']
        # get page number of first usable text
        figure_texts = self.get_figure_texts(json_data)
        main_text_index = 0
        for figure_dict in figure_texts:
            figure_page_num = figure_dict['prov'][0]['page']
            while (main_text_index < len(json_data['main-text'])) and (
                    (json_data['main-text'][main_text_index]['type'] in self.text_types_skip) or
                    (json_data['main-text'][main_text_index]['prov'][0]['page'] < figure_page_num)):
                main_text_index += 1
            json_data['main-text'].insert(main_text_index, figure_dict)
        main_text_index = 0
        while main_text_index < len(json_data['main-text']):
            if json_data['main-text'][main_text_index]['type'] not in self.text_types_skip:
                text = re.sub('[^\u00C0-\u017FA-Za-z0-9"!@#$%^&*()\[\]{};:,./<>?\\\|`~\-=_+/s]+', ' ',
                           json_data['main-text'][main_text_index]['text'])
                page = json_data['main-text'][main_text_index]['prov'][0]['page']
                if page not in page_texts_dict.keys():
                    page_texts_dict[page] = []
                page_texts_dict[page].append({'text': text,
                                              'bbox': json_data['main-text'][main_text_index]['prov'][0]['bbox']})
            main_text_index += 1
        page_figures_dict = {}
        figures_index = 0
        while figures_index < len(json_data['figures']):
            if json_data['figures'][figures_index]['type'] != 'picture':
                raise NotImplementedError("Code only support figures of type picture")
            page_number = json_data['figures'][figures_index]['prov'][0]['page']
            if json_data['figures'][figures_index]['confidence'] >= min_fig_conf:
                page_figure_bbox = json_data['figures'][figures_index]['prov'][0]['bbox']  # [left, bottom, right, top] bottom left is 0,0
                if page_figure_bbox[0] > page_figure_bbox[2] or page_figure_bbox[1] > page_figure_bbox[3]:
                    figures_index += 1
                    print(f'Bad bbox {page_figure_bbox} in pdf: {pdf_basename} page {page_number}, skipping')
                    continue
                page_dim = self.get_page_dims(page_number)
                page_area = page_dim['height'] * page_dim['width']
                figure_area = (page_figure_bbox[2] - page_figure_bbox[0]) * (page_figure_bbox[3] - page_figure_bbox[1])
                figure_page_area_ratio = figure_area / page_area
                if figure_page_area_ratio < 0.02:
                    figures_index += 1
                    print(
                        f'figure is less than 2 percent of the page in pdf: {pdf_basename} page {page_number}, skipping')
                    continue
                if figure_page_area_ratio > 0.6:
                    figures_index += 1
                    print(
                        f'figure is more than 60 percent of the page in pdf: {pdf_basename} page {page_number}, skipping')
                    continue
                if page_number not in page_figures_dict.keys():
                    page_figures_dict[page_number] = []
                page_figures_dict[page_number].append({'bbox': json_data['figures'][figures_index]['prov'][0]['bbox']})
            else:
                print(f'Image in pdf: {pdf_basename} page {page_number}, has confidence lower than {min_fig_conf}, skipping image')
            figures_index += 1
        self.text_data = page_texts_dict
        self.figures_data = page_figures_dict

    def get_first_page_number(self):
       return min(self.data.keys())

    def get_page_texts(self, page_number):
        return self.data[page_number]

    def get_page_dims(self, page_number):
        return self.json_data['page-dimensions'][page_number - 1]

    def get_figure_texts(self, json_data):
        text_dicts = []
        for figure_dict in json_data['figures']:
            if figure_dict['cells']['data']:
                data_list = figure_dict['cells']['data']
                page_text_list = [text_data_list[-1] for text_data_list in data_list if len(text_data_list[-1]) > 4]
                text_boxes = [text_data_list[:4] for text_data_list in data_list if len(text_data_list[-1]) > 4]
                if not text_boxes:
                    continue
                page_index = figure_dict['prov'][0]['page']
                page_dim = json_data['page-dimensions'][page_index-1]
                connected_comp_list = self.get_connected_components(text_boxes, (page_dim['width'], page_dim['height']),
                                                                    add_perc=0.001)
                merged_text_list = self.merge_text_dicts(connected_comp_list, text_boxes, page_text_list)
                for merged_text_dict in merged_text_list:
                    main_text_dict = {'name': 'text', 'type': 'paragraph', 'text':  merged_text_dict['text'],
                                      'prov': [{'bbox': merged_text_dict['bbox'], 'page': page_index}]}
                    text_dicts.append(main_text_dict)
        return text_dicts

    def merge_text_dicts(self, connected_comp_list, boxes, texts):
        new_texts_list = []
        for con_comp in connected_comp_list:
            con_boxes = [boxes[i] for i in con_comp]
            merged_box = self.merge_boxes(con_boxes)
            con_texts = [texts[i] for i in con_comp]
            merged_text = self.merge_texts(con_boxes, con_texts)
            new_texts_list.append({'bbox': merged_box, 'text': merged_text})
        return new_texts_list

    @staticmethod
    def merge_texts(boxes, texts):
        # x0, y0, x1, y1; (left, bottom, right, top) ; 0,0 is bottom left
        if len(texts) == 1:
            return texts[0]
        struct_array = np.array([(box[2], box[3]) for box in boxes], dtype=[("right", float), ("top", float)])
        sort_indices = np.argsort(struct_array, order=['top', 'right'])[::-1]
        return ' '.join([texts[i] for i in sort_indices])

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

    def get_connected_components(self, txt_boxes, page_size, add_perc=0.05):
        # x0, y0, x1, y1; (left, bottom, right, top) ; 0,0 is bottom left
        boxes = np.stack(txt_boxes)
        # boxes_percentile_vert = add_perc * (boxes[:, 3] - boxes[:, 1])
        boxes_percentile_vert = add_perc * page_size[1]
        boxes_percentile_horiz = boxes_percentile_vert / 4
        # boxes_percentile_horiz = add_perc * (boxes[:, 2] - boxes[:, 0])
        boxes[:, 0] = np.maximum(np.zeros_like(boxes[:, 0]), boxes[:, 0] - boxes_percentile_horiz)
        boxes[:, 2] = np.minimum(np.ones_like(boxes[:, 2]) * page_size[0], boxes[:, 2] + boxes_percentile_horiz)
        boxes[:, 1] = np.maximum(np.zeros_like(boxes[:, 1]), boxes[:, 1] - boxes_percentile_vert)
        boxes[:, 3] = np.minimum(np.ones_like(boxes[:, 3]) * page_size[1], boxes[:, 3] + boxes_percentile_vert)
        # get connected components if boxes are intersecting
        all_combs = list(itertools.combinations(np.arange(boxes.shape[0]), 2))
        g = Graph(boxes.shape[0])
        for i, j in all_combs:
            is_inter = self.is_intersecting(boxes[i, :], boxes[j, :])
            if is_inter:
                g.addEdge(i, j)
        cc, cc_dict = g.connectedComponents()
        return cc

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


class Graph:

    # init function to declare class variables
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]

    def DFSUtil(self, temp, v, visited):

        # Mark the current vertex as visited
        visited[v] = True

        # Store the vertex to list
        temp.append(v)

        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:

                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp

    # method to add an undirected edge
    def addEdge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)

    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(self):
        visited = []
        cc = []
        cc_dict = {}
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                connected_comp = self.DFSUtil(temp, v, visited)
                cc.append(connected_comp)
                for vert in connected_comp:
                    if vert not in cc_dict.keys():
                        cc_dict[vert] = connected_comp
        return cc, cc_dict


def main():
    args = parse_arguments()
    if args.debug:
        pydevd_pycharm.settrace('9.171.89.17', port=55551, stdoutToServer=True, stderrToServer=True, suspend=False)
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    gen = DS_to_pkl(args)
    if args.mode == 'default':
        success = gen.generate_default(min_fig_conf=0.0)
    elif args.mode == 'car_manuals':
        success = gen.generate_car_manuals(min_fig_conf=0.0)
    else:
        raise NotImplementedError(f'Mode {args.mode} not supported.')
    if success:
        print("Finished successfully")


if __name__ == '__main__':
    main()
