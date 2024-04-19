import torch
from model import Model
import os
model = Model(device = "cuda:0", dtype = torch.float16)

prompts = list(open("prompts.txt"))
prompts = [i.strip() for i in prompts]
print(prompts)
seeds = [0,42,2333,530,9,1184, 252, 457, 431, 713]
from hf_utils import get_model_list
model_list = get_model_list()
for prompt in prompts:
  for seed in seeds:
    params = {"t0": 44, "t1": 47 , "seed":seed,"motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 16}
    out_dir = f"output_8fps_16len/seed_{seed}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path, fps = os.path.join(out_dir,f"{prompt.replace(' ','_')}_seed{seed}.mp4"), 8
    # model.process_text2video(prompt, fps = fps, path = out_path, **params)

    # for idx, name in enumerate(model_list):
    #   print(idx, name)
    # idx = int(input("Select the model by the listed number: ")) # select the model of your choice
    model.process_text2video(prompt, model_name = model_list[0], fps = fps, path = out_path, **params)