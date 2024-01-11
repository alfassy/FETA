#!/bin/sh
# Vanilla CLIP many shot
python -m torch.distributed.launch --nproc_per_node $GPU_NUM -m training.main --exp_mode zero --fold 2 --distributed --data_mode cm --name vanilla_clip_many_shot_2 --batch-size 32 --lr 5e-05 --epochs 20 --model RN50 --logs $LOGS_FOLDER --val-data $DATA_ROOT/car_manuals_data/car_manuals_data.pkl --openai-pretrained --save-frequency 50
# Average results
python training/utils.py --logs_path $LOGS_FOLDER --run_name vanilla_clip_many_shot --exp_mode many