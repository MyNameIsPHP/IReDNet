# Iterative-Recurrent-Densely-Connected-Convolutional-Network
## Installation
Run the script below to install (Conda):
```
conda create -n irednet python=3.6.7
conda activate irednet
conda install -y pytorch=1.4.0 cudatoolkit=10.0 torchvision -c pytorch
conda install h5py opencv
pip install tensorboardX scikit-image==0.17.2
```

## Datasets
To train and evaluate the models, please download training and testing datasets from 
https://drive.google.com/file/d/1ktuaInh9Lsnxxdml8f3x4NASeZe6I0Gc/view?usp=sharing
and place the unzipped folders into the project folder.

## Getting Started

### 1) Testing
Run shell scripts to test the models:
```bash
bash test.sh  
```

### 2) Training

Run shell scripts to train the models:
```bash
bash train.sh      
```
The models are saved into `./logs/` folder by default



