import torch
from sklearn.neighbors import NearestNeighbors
import os
import numpy as np
import pickle


def calc_nn_feta(search_features, query_features, search_labels=None, query_labels=None, metric='correlation',
                 i2t=False, output_path=None, image_paths=None):
    n_neighbors = 10
    acc1, acc5, acc10 = [], [], []
    doc_path = []
    for doc_key in search_features.keys():
        doc_path.append(os.path.dirname(image_paths[doc_key][0]))
        cls = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=-1,
                               metric=metric).fit(search_features[doc_key])
        _, top_n_matches_ids = cls.kneighbors(query_features[doc_key])  # num_images x n_neighbors
        if i2t:
            search_labels = query_labels[doc_key].transpose()  # num_texts x num_images
            search_labels_proj = np.expand_dims(np.arange(query_features[doc_key].shape[0]), axis=1).repeat(n_neighbors,
                                                                                                            axis=1)  # num_images x n_neighbors
            correct = search_labels[top_n_matches_ids, search_labels_proj]
            acc1.append(correct[:, 0:1].any(-1).mean())
            acc5.append(correct[:, 0:5].any(-1).mean())
            acc10.append(correct[:, 0:10].any(-1).mean())
        else:
            acc1_images, acc5_images, acc10_images = [], [], []
            for image_index in range(search_features[doc_key].shape[0]):
                suitable_texts_ind = np.where(search_labels[doc_key][image_index, :] == 1)
                ranks = np.where((top_n_matches_ids[suitable_texts_ind, :] == image_index).squeeze())
                if len(ranks) == 1:
                    if ranks[0].size == 0:
                        rank = 10000
                    else:
                        rank = ranks[0][0]
                elif ranks[0].size == 0:
                    rank = 10000
                else:
                    rank = ranks[1].min()
                    closest_text_ind = ranks[0][ranks[1].argmin()]
                acc1_images.append(int(rank < 1))
                acc5_images.append(int(rank < 5))
                acc10_images.append(int(rank < 10))
            acc1.append(sum(acc1_images) / len(acc1_images))
            acc5.append(sum(acc5_images) / len(acc5_images))
            acc10.append(sum(acc10_images) / len(acc10_images))
    if output_path is not None:
        acc_dict = {'acc1': acc1, 'acc5': acc5, 'acc10': acc10, 'doc_paths': doc_path}
        with open(output_path+['_t2i', '_i2t'][i2t]+'_acc.pkl', 'wb') as out_file:
            pickle.dump(acc_dict, out_file, protocol=pickle.HIGHEST_PROTOCOL)
    result_dict = {'acc1': sum(acc1) / len(acc1), 'acc5': sum(acc5) / len(acc5), 'acc10': sum(acc10) / len(acc10)}
    return result_dict


def get_features_feta(model, data_loader, args=None):
    image_features_per_doc = {}
    image_paths_per_doc = {}
    text_features_per_doc = {}
    texts_per_doc = {}
    with torch.no_grad():
        # reset count dictionaries
        data_loader.dataset.prep_val_data()
        for batch_data in data_loader:
            (image_inputs, text_inputs, image_labels, text_labels, img_paths, texts) = batch_data
            image_inputs, text_inputs = image_inputs.cuda(), text_inputs.cuda()
            image_features, text_features, logit_scale = model(image_inputs, text_inputs)
            batch_doc_split_image = [0] + [i for i in range(1, len(image_labels)) if image_labels[i] != image_labels[i-1]] + [len(image_labels)]
            for split_index in range(1, len(batch_doc_split_image)):
                prev_index = batch_doc_split_image[split_index-1]
                change_index = batch_doc_split_image[split_index]
                doc_num = int(image_labels[prev_index])
                if doc_num not in image_features_per_doc.keys():
                    image_features_per_doc[doc_num] = image_features[prev_index:change_index, :].detach().cpu().clone()
                    image_paths_per_doc[doc_num] = []
                else:
                    image_features_per_doc[doc_num] = torch.cat((image_features_per_doc[doc_num],
                                                                 image_features[prev_index:change_index, :].detach().cpu().clone()), dim=0)
                image_paths_per_doc[doc_num].extend(img_paths[prev_index:change_index])
            batch_doc_split_text = [0] + [i for i in range(1, len(text_labels)) if text_labels[i] != text_labels[i-1]] + [len(text_labels)]
            for split_index in range(1, len(batch_doc_split_text)):
                prev_index = batch_doc_split_text[split_index-1]
                change_index = batch_doc_split_text[split_index]
                doc_num = int(text_labels[prev_index])
                if doc_num not in text_features_per_doc.keys():
                    text_features_per_doc[doc_num] = text_features[prev_index:change_index, :].detach().cpu().clone()
                    texts_per_doc[doc_num] = []
                else:
                    text_features_per_doc[doc_num] = torch.cat((text_features_per_doc[doc_num],
                                                                text_features[prev_index:change_index, :].detach().cpu().clone()), dim=0)
                texts_per_doc[doc_num].extend(texts[prev_index:change_index])
    if args.data_mode == 'cm':
        image_labels = data_loader.dataset.gen_sim_val_gt(image_paths_per_doc)
    else:
        image_labels = data_loader.dataset.gt_matrix
    return image_features_per_doc, text_features_per_doc, image_labels, None, image_paths_per_doc, texts_per_doc
