# Instructions for running FETA on custom data
## Data process
### Automatic PDFs processing pipeline
Get DeepSearch username and api_key as explained here: https://ds4sd.github.io/deepsearch-toolkit/getting_started/  
Supported input types are explained here: https://github.com/DS4SD/deepsearch-examples/blob/main/examples/document_conversion/notebooks/convert_documents.ipynb  
Note that DeepSearch engine is evolving all the time so below run command will return different outputs with time.
```bash
python ocr_utils/run_deep_search.py --source-path <path_to_input_pdfs> --username <DS_username> --api_key <key> --target-dir <path_for_output>
```
### Subsequent data processing
```bash
export DATA_NAME=custom_data
export TARGET_DIR=../FETA_data/custom_data
python ocr_utils/feta_post_processing.py --source-dir <inputs_dir> --target-dir $TARGET_DIR --name $DATA_NAME --tmp_dir <path_to_save_visualizations>
```
source-dir: A single folder with pdfs + jsons acquired in previous stage (unzipped)  
target-dir: cropped images and pkl file will be created here.  
Create fold train-test split function and add to src/training/data.py line 35's if clause. Update data_mode, exp_mode
and fold parameters to fit your data.  
feta_post_processing.py might not work as DeepSearch constantly develop, please open an issue on Github if that happens.
## Train
Set required environment variables:
```bash
export GPU_NUM=1
export LOGS_FOLDER=../results
export DATA_MODE=default
export EXP_MODE=default
export FOLD=0
export RUN_NAME=custom_data
source commands/train_your_data.sh
```
DATA_MODE is set to default, if you want your own FOLD train-test split, create a function and add to
src/training/data.py line 35's if clause. Update DATA_MODE and add to src/training/data.py line 35's if clause.
Add EXP_MODE, and FOLD parameters to your above function if needed.
train-data and val-data params should point to the pkl file created by feta post-processing.  
Outputs will be saved under {args.logs}/{args.name}/ ($LOGS_FOLDER/$RUN_NAME)