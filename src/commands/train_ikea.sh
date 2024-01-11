#!/bin/sh
python -m torch.distributed.launch --nproc_per_node $GPU_NUM -m training.main --exp_mode many --fold 2 --distributed --all_page_texts --data_mode default --name FETA_IKEA_2 --batch-size 64 --lr 5e-05 --epochs 20 --model RN50 --logs $LOGS_FOLDER --train-data $DATA_ROOT/ikea_data/ikea_data.pkl --val-data $DATA_ROOT/ikea_data/ikea_data.pkl --openai-pretrained --save-frequency 50
# Average results
python training/utils.py --logs_path $LOGS_FOLDER --run_name FETA_IKEA --exp_mode ikea