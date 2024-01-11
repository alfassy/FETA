#!/bin/sh
# data_mode is set to default, if you want your own fold train-test split, create a function and add to
# src/training/data.py line 35's if clause. Update data_mode and add to src/training/data.py line 35's if clause.
# Add exp_mode, and fold parameters to your above function if needed.
# Outputs will be saved under {args.logs}/{args.name}/
# FETA train on your own data
python -m torch.distributed.launch --nproc_per_node $GPU_NUM -m training.main --exp_mode $EXP_MODE --all_page_texts --fold $FOLD --distributed --data_mode $DATA_MODE --name $RUN_NAME --batch-size 32 --lr 5e-05 --epochs 20 --model RN50 --logs $LOGS_FOLDER --train-data $TARGET_DIR/$DATA_NAME.pkl --val-data $TARGET_DIR/$DATA_NAME.pkl --openai-pretrained --save-frequency 50

# FETA test on your own data (only provide val-data param and not train-data param
python -m torch.distributed.launch --nproc_per_node $GPU_NUM -m training.main --exp_mode $EXP_MODE --all_page_texts --fold $FOLD --distributed --data_mode $DATA_MODE --name $RUN_NAME --batch-size 32 --lr 5e-05 --epochs 20 --model RN50 --logs $LOGS_FOLDER --val-data $TARGET_DIR/$DATA_NAME.pkl --openai-pretrained --save-frequency 50
