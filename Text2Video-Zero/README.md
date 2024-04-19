# Text2Video-Zero
A text to video method

- paper: [Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators](https://arxiv.org/abs/2303.13439)

- code: https://github.com/Picsart-AI-Research/Text2Video-Zero

## Install

---
1. clone the source repo
```
git clone https://github.com/Picsart-AI-Research/Text2Video-Zero.git
cd Text2Video-Zero/
```
2. create env and install package
    
    python 3.9
    - refer to the source repo
        ```
        virtualenv --system-site-packages -p python3.9 venv
        source venv/bin/activate
        pip install -r requirements.txt
      ```
      conda env is ok, not a must to use virtualenv
    - For convenience, you could just import the conda env using `environment.yaml`.
## Usage

---
- `inference.py` is the implemented script refer to the source repo.
- `inference_all.py` is an implement of generating videos according to the input prompt list file.