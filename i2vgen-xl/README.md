# I2vgen-XL
[code](https://github.com/ali-vilab/VGen);
[paper](https://arxiv.org/abs/2311.04145);
[source](https://modelscope.cn/models/iic/i2vgen-xl/summary)

I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models

An image-to-video method, including two steps:
1. image2video
2. video2video: generating high-quality videos

Those two steps could be used seperately, especially the steps 2.
## Install

---
Create a conda environment and install the required package.
```
conda create -n vgen python=3.8
conda activate vgen
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- option 1: download the `requirment.txt` from original [repo](https://github.com/ali-vilab/VGen) and follow the former instuction
- option 2: import the conda environment from my created `vgen.yaml`
## Usage

---
Refer to the scripts and modify the path to generate your videos.

- `mycode.py`
    
    generate videos according to the demo images and prompts from the original paper demo.
- `image2videos_2step.py` 

    This file produce the videos of 4s of the input images.
    input: image dir; text prompt
    two steps: 
        step1: image2video pipeline. You can finish here if the videos is satisfying.
        step2: video2video pipeline to get a high quality videos.
