# Download models
To download the models you will need 14 free GB
```bash
export MODELS_ROOT=../pt_FETA_models
cd $MODELS_ROOT
wget https://huggingface.co/datasets/alfassy/FETA_IKEA_CAR_MANUALS/resolve/main/feta_pt_models.tar.gz
tar -xvf feta_pt_models.tar.gz
# After untar you can safely delete the tar.
rm feta_pt_models.tar.gz
cd .../src
```
# models files structure
.  
├── cm_zero  
├── cm_one  
├── cm_few   
├── cm_many  
└── ikea



