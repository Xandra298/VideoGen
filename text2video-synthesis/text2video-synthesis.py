from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import torch
import os
# seed = 0
num_inference_steps= 25
num_frames = 16
import random
import numpy as np

# get list of prompts
prompts = list(open("prompts.txt"))
prompts = [i.strip() for i in prompts]
print(prompts)

#seed
# seed = random.sample(list(np.arange(0,2000)),5)
seeds = [0,42,2333,530,9,1184, 252, 457, 431, 713] # just for test

# define pipeline:text-to-video-synthesis
p = pipeline('text-to-video-synthesis','damo/text-to-video-synthesis',torch_dtype=torch.float16,
                                         variant='fp16')

# generating for all
print("start generating...")
cout = 0
for prompt in prompts:
    cout+=1
    test_text = {
        'text': prompt,
    }
    # for each seed
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        if len(prompt) < 35:
            name = prompt.split('.')[0]
        else: name = prompt[:35]
        output_path = f"output/seed_{seed}"
        if not os.path.exists(output_path):
                os.makedirs(output_path)
        # generate, save at output_path
        output_video_path = p(test_text,output_video=os.path.join(output_path,f"{name}_seed{seed}.mp4"),num_inference_steps=num_inference_steps,
                        num_frames=num_frames,
                        generator=generator)[OutputKeys.OUTPUT_VIDEO]
        print(f'finish generating:{prompt}, seed:{seed},output_video path: {output_video_path}')
print(f"fininsh generating prompts: {cout}.")