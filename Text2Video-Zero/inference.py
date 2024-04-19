import torch
from model import Model

model = Model(device = "cuda:1", dtype = torch.float16)

prompt = "A panda surfing on a wakeboard."
params = {"t0": 44, "t1": 47 , "seed":42,"motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 8}

out_path, fps = f"./videos/text2video_{prompt.replace(' ','_')}.mp4", 4
# model.process_text2video(prompt, fps = fps, path = out_path, **params)
from hf_utils import get_model_list
model_list = get_model_list()
for idx, name in enumerate(model_list):
  print(idx, name)
idx = int(input("Select the model by the listed number: ")) # select the model of your choice
model.process_text2video(prompt, model_name = model_list[idx], fps = fps, path = out_path, **params)