# Installation

This codebase is tested on Ubuntu 18.04.6 LTS with Python 3.8. Follow the below steps to create the environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -n hpt python=3.8

# Activate the environment
conda activate hpt

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

* Clone HPT code repository and install requirements
```bash
# Clone HPT code base
git clone https://github.com/Vill-Lab/2024-AAAI-HPT.git

cd 2024-AAAI-HPT/
# Install requirements

pip install -r requirements.txt
```
