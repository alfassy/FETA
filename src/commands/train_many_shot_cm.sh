#!/bin/sh
# FETA many shot - run only on Nissan which has more training documents than other manufacturers and is thus considered many shot
python -m torch.distributed.launch --nproc_per_node $GPU_NUM -m training.main --exp_mode many --fold 2 --distributed --data_mode cm --name feta_many_shot_2 --batch-size 32 --lr 5e-05 --epochs 20 --model RN50 --logs $LOGS_FOLDER --train-data $DATA_ROOT/car_manuals_data/car_manuals_data.pkl --val-data $DATA_ROOT/car_manuals_data/car_manuals_data.pkl --openai-pretrained --save-frequency 50
# Average results
python training/utils.py --logs_path $LOGS_FOLDER --run_name feta_many_shot --exp_mode many