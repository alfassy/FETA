# Download data
To download this data you will need 7 free GB
```bash
export DATA_ROOT=../FETA_data
cd $DATA_ROOT
wget https://huggingface.co/datasets/alfassy/FETA_IKEA_CAR_MANUALS/resolve/main/feta_data.tar.gz
tar -xvf feta_data.tar.gz
# After untar you can safely delete the tar.
rm feta_data.tar.gz
cd ../src
```
# Data files structure
.  
├── car_manuals_data  
│   ├── images  
│   ├── car_manuals_data.pkl  
│   └── texts  
├── ikea_data  
│   ├── ikea_data.pkl  
│   ├── images  
│   └── texts
# Data's texts and annotations used in training
Texts and annotations of the entire data, page and image are available in the pkl file which is located in the main directory of each dataset.
# Text and annotations data
Inside each tsv file, which can be opened in Excel or as text file, there are four column listing the texts and annotations of the page: page_number, text_ind, text, bbox.  
tsv files aren't currently supported for training, please use pkl files.
