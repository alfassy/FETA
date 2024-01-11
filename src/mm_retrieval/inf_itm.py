import os
from time import time
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
import logging


def default_get_features(model, data_loader):
    all_image_features = None
    for batch_data in data_loader:
        (image_inputs, text_inputs, img_labels, txt_labels) = batch_data
        image_inputs, text_inputs = image_inputs.cuda(), text_inputs.cuda()
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if all_image_features is None:
            all_image_features = image_features
            all_text_features = text_features
            all_img_labels = img_labels
            all_text_labels = txt_labels
        else:
            all_image_features = torch.cat((all_image_features, image_features), dim=0)
            all_text_features = torch.cat((all_text_features, text_features), dim=0)
            all_img_labels = torch.cat((all_img_labels, img_labels), dim=0)
            all_text_labels = torch.cat((all_text_labels, txt_labels), dim=0)
    return all_image_features, all_text_features, all_img_labels, all_text_labels, None, None


def calc_nn(search_features, query_features, search_labels=None, query_labels=None, metric='correlation'):
    search_labels, query_labels = search_labels.numpy(), query_labels.numpy()
    search_features, query_features = search_features.cpu(), query_features.cpu()
    n_neighbors = 10
    cls = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=-1, metric=metric).fit(search_features)
    _, top_n_matches_ids = cls.kneighbors(query_features)
    if search_labels is None:
        top_n_labels = top_n_matches_ids
    else:
        top_n_labels = search_labels[top_n_matches_ids]
    correct = query_labels[:, None] == top_n_labels
    acc1 = correct[:, 0:1].any(-1).mean()
    acc5 = correct[:, 0:5].any(-1).mean()
    acc10 = correct[:, 0:10].any(-1).mean()
    return acc1, acc5, acc10


def image_text_ret(model=None, eval_dataloader=None, get_features=default_get_features, calc_nn_func=calc_nn,
                   dataset='coco', eval_split='1k', nn_metric='correlation', val_preprocess=None, output_path=None,
                   tb_writer=None, is_master=True, epoch=0, args=None):
    '''
    Run image to text and text to image retrieval and print results.
    The default behavior uses CLIP model, eval_dataloader and the default_get_features def which was created for CLIP.
    CLIP is used as an example, to use your own model provide model, eval_dataloader and get_features def.
    :param model: model to use with eval_dataloader
    :param eval_dataloader: dataloader which works with your model
    :param get_features: a function that receives the appropriate model and eval_dataloader and returns all test
    image features, all test text features and appropriate labels of image to txt compatability.
    :param calc_nn_func: a function that receives the output of the get_features function and calculate NN i2t t2i retrieval.
    :param dataset: either coco/ flickr30k
    :param eval_split: only used for coco dataset, either 1k/ 5k.
    :param nn_metric: sklearn NearestNeighbors distance metric.
    :param val_preprocess: preprocess transform for validation dataset.
    :param output_path: path to save output accuracy file
    several texts.
        options: (braycurtis/canberra/chebyshev/correlation/dice/hamming/jaccard/kulsinski/mahalanobis/matching/
        /minkowski/rogerstanimoto/russellrao/seuclidean/sokalmichener/sokalsneath/sqeuclidean/yule)
    :return: nothing.
    '''
    if not is_master:
        return
    pre_loader = False
    if eval_dataloader is None or model is None:
        import mm_retrieval.CLIP.clip as clip
        from mm_retrieval.CLIP.data_loader_clip import LmdbPairDatasetCLIP
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is None:
            model, clip_preprocess = clip.load('RN50', device)
            model.cuda()
            pre_loader = True
        elif val_preprocess is None:
            raise NotImplementedError('val_preprocess required with model')
        else:
            clip_preprocess = val_preprocess
        if eval_dataloader is None:
            LMDB_DATA_ROOT = os.environ.get('LMDB_DATA_ROOT', '/dccstor/aarbelle1/data/PhraseGrounding/data')
            lmdb_data_path = (os.path.join(LMDB_DATA_ROOT, dataset),)
            eval_dataset = LmdbPairDatasetCLIP(lmdb_data_path, 'val', max_len=100, clip_tokenizer=clip.tokenize,
                                               clip_preprocess=clip_preprocess, eval_data_split=dataset+eval_split)
            eval_dataloader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1000, shuffle=False,
                                                          collate_fn=eval_dataset.collate_fn(), num_workers=0)
        else:
            if pre_loader:
                eval_dataloader.dataset.clip_preprocess = clip_preprocess
    model.eval()
    st = time()
    print("start running Image/Text Retrieval evaluation ...")
    logging.info("start running Image/Text Retrieval evaluation ...")
    t2i_a1, i2t_a1 = inference_itm(model, eval_dataloader, get_features, calc_nn_func, nn_metric, output_path, tb_writer=tb_writer, epoch=epoch, args=args)
    tot_time = time()-st
    print(f"evaluation finished in {int(tot_time)} seconds, ")
    logging.info(f"evaluation finished in {int(tot_time)} seconds, ")
    return t2i_a1, i2t_a1


def inference_itm(model, eval_loader, get_features, calc_nn_func, nn_metric, output_path, tb_writer=None, epoch=0, args=None):
    model.eval()
    all_image_features, all_text_features, all_image_labels, all_txt_labels, image_paths_per_doc, texts_per_doc = get_features(model, eval_loader, args=args)
    # txt2img acc
    t2i_res = calc_nn_func(all_image_features, all_text_features, search_labels=all_image_labels,
                           query_labels=all_txt_labels, metric=nn_metric, i2t=False, output_path=output_path,
                           image_paths=image_paths_per_doc)
    t2i_a1, t2i_a5, t2i_a10 = t2i_res['acc1'], t2i_res['acc5'], t2i_res['acc10']
    print(f'Txt to img acc 1,5,10: {t2i_a1}, {t2i_a5}, {t2i_a10}')
    logging.info(f'Txt to img acc 1,5,10: {t2i_a1}, {t2i_a5}, {t2i_a10}')
    if tb_writer is not None:
        for name, val in t2i_res.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"Test/t2i_ret_{name}", val, epoch)
    # img2txt acc
    i2t_res = calc_nn_func(all_text_features, all_image_features, search_labels=all_txt_labels,
                           query_labels=all_image_labels, metric=nn_metric, i2t=True, output_path=output_path,
                           image_paths=image_paths_per_doc)
    i2t_a1, i2t_a5, i2t_a10 = i2t_res['acc1'], i2t_res['acc5'], i2t_res['acc10']
    print(f'Img to txt acc 1,5,10: {i2t_a1}, {i2t_a5}, {i2t_a10}')
    logging.info(f'Img to txt acc 1,5,10: {i2t_a1}, {i2t_a5}, {i2t_a10}')
    if tb_writer is not None:
        for name, val in i2t_res.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"Test/i2t_{name}", val, epoch)
    return t2i_a1, i2t_a1
