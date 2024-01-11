#!/bin/sh
# Vanilla CLIP zero shot
python -m torch.distributed.launch --nproc_per_node $GPU_NUM -m training.main --exp_mode zero --fold 0 --distributed --data_mode cm --name vanilla_clip_zero_shot_0 --batch-size 32 --lr 5e-05 --epochs 20 --model RN50 --logs $LOGS_FOLDER --val-data $DATA_ROOT/car_manuals_data/car_manuals_data.pkl --openai-pretrained --save-frequency 50
python -m torch.distributed.launch --nproc_per_node $GPU_NUM -m training.main --exp_mode zero --fold 1 --distributed --data_mode cm --name vanilla_clip_zero_shot_1 --batch-size 32 --lr 5e-05 --epochs 20 --model RN50 --logs $LOGS_FOLDER --val-data $DATA_ROOT/car_manuals_data/car_manuals_data.pkl --openai-pretrained --save-frequency 50
python -m torch.distributed.launch --nproc_per_node $GPU_NUM -m training.main --exp_mode zero --fold 2 --distributed --data_mode cm --name vanilla_clip_zero_shot_2 --batch-size 32 --lr 5e-05 --epochs 20 --model RN50 --logs $LOGS_FOLDER --val-data $DATA_ROOT/car_manuals_data/car_manuals_data.pkl --openai-pretrained --save-frequency 50
python -m torch.distributed.launch --nproc_per_node $GPU_NUM -m training.main --exp_mode zero --fold 3 --distributed --data_mode cm --name vanilla_clip_zero_shot_3 --batch-size 32 --lr 5e-05 --epochs 20 --model RN50 --logs $LOGS_FOLDER --val-data $DATA_ROOT/car_manuals_data/car_manuals_data.pkl --openai-pretrained --save-frequency 50
python -m torch.distributed.launch --nproc_per_node $GPU_NUM -m training.main --exp_mode zero --fold 4 --distributed --data_mode cm --name vanilla_clip_zero_shot_4 --batch-size 32 --lr 5e-05 --epochs 20 --model RN50 --logs $LOGS_FOLDER --val-data $DATA_ROOT/car_manuals_data/car_manuals_data.pkl --openai-pretrained --save-frequency 50
# Average results
python training/utils.py --logs_path $LOGS_FOLDER --run_name vanilla_clip_zero_shot --exp_mode zero
