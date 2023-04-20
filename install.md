# 1. Installment

### 1.1 Download Pretrained Model
First cd in the project path, then download the pretrained model from grit.

```bash
cd [YOUR_PATH_TO_THIS_PROJECT]
mkdir checkpoints
cd checkpoints
wget -c https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth
cd ..
```

### 1.2 Install Enviornment

Run

```
conda create -n -y vlog python=3.8
conda activate vlog
pip install -r requirements.txt
```
