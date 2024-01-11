This is the code for paper "FETA: Towards Specializing Foundation Models for
Expert Task Applications",    
It was published as a main conference paper in NeurIPS 2022.  
arxiv link: https://arxiv.org/abs/2209.03648   
Papers with code link: https://paperswithcode.com/paper/feta-towards-specializing-foundation-models  
Car manuals benchmark in papers with code: https://paperswithcode.com/dataset/feta-car-manuals  
IKEA benchmark in papers with code:https://paperswithcode.com/dataset/feta-ikea  
If you use this code, please cite the following bibtex: 
```
@misc{https://doi.org/10.48550/arxiv.2209.03648, doi = {10.48550/ARXIV.2209.03648},url = {https://arxiv.org/abs/2209.03648},author = {Alfassy, Amit and Arbelle, Assaf and Halimi, Oshri and Harary, Sivan and Herzig, Roei and Schwartz, Eli and Panda, Rameswar and Dolfi, Michele and Auer, Christoph and Saenko, Kate and Staar, PeterW. J. and Feris, Rogerio and Karlinsky, Leonid},keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},title = {FETA: Towards Specializing Foundation Models for Expert Task Applications},publisher = {arXiv}, year = {2022},copyright = {arXiv.org perpetual, non-exclusive license}}
```
![alt text](https://github.com/alfassy/FETA/blob/main/FETA_data/main_figure.png?raw=true)  
This code repository is based on open-clip: https://github.com/mlfoundations/open_clip
# Installation
```bash
git clone  
conda create -n feta python=3.8  
conda activate feta  
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
conda env update --name feta --file environment.yml  
cd feta/src
```
# Run
## Data preparation 
Refer to [README_data.md](README_data.md) for data download and preparation instructions
## FETA trained models preparation 
Refer to [README_models.md](README_models.md) for FETA trained models download and preparation instructions.
## Run FETA on your own data
Refer to [README_run_on_custom_data.md](README_run_on_custom_data.md) for instructions.
## Reproduce paper results
Set required environment variables:
```bash
export GPU_NUM=1
export LOGS_FOLDER=../results
export DATA_ROOT=../FETA_data
export MODELS_ROOT=../pt_FETA_models
```
Outputs will be saved under {args.logs}/{args.name}/
### Run FETA test with our pretrained models
Zero shot Car-Manuals
```bash
source commands/test_zero_shot_cm.sh
```
One shot Car-Manuals
```bash
source commands/test_one_shot_cm.sh
```
Few shot Car-Manuals
```bash
source commands/test_few_shot_cm.sh
```
Many shot Car-Manuals
```bash
source commands/test_many_shot_cm.sh
```
IKEA many shot
```bash
source commands/test_ikea.sh
```
### Run test with clip400M pretrained model.
Zero shot Car-Manuals
```bash
source commands/test_pt_clip_zero_cm.sh
```
One shot Car-Manuals
```bash
source commands/test_pt_clip_one_cm.sh
```
Few shot Car-Manuals
```bash
source commands/test_pt_clip_few_cm.sh
```
Many shot Car-Manuals
```bash
source commands/test_pt_clip_many_cm.sh
```
IKEA many shot
```bash
source commands/test_ikea_pt_clip.sh
```
### Train FETA models to reproduce tables 13 and 14 from Arxiv's version.
Use the below commands to reproduce tables 13 and 14 from Arxiv's version.

Zero shot Car-Manuals
```bash
source commands/train_zero_shot_cm.sh
```
One shot Car-Manuals
```bash
source commands/train_one_shot_cm.sh
```
Few shot Car-Manuals
```bash
source commands/train_few_shot_cm.sh
```
Many shot Car-Manuals
```bash
source commands/train_many_shot_cm.sh
```
IKEA many shot
```bash
source commands/train_ikea.sh
```

