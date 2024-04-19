from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
# import torch
import os
# get list of prompts
prompts = list(open("../text2video-synthesis/prompts.txt"))
prompts = [i.strip() for i in prompts]
print(prompts)

#seed
# seed = random.sample(list(np.arange(0,2000)),5)
seeds = [0,42,2333,530,9,1184, 252, 457, 431, 713]
video_to_video_pipe = pipeline(task="video-to-video", model='damo/Video-to-Video', model_revision='v1.1.0', device='cuda:0')
print("start fining videos...")
for prompt in prompts:
    for seed in seeds:
        ##text2video-synthesis
        # in_dir = f"../text2video-synthesis/output/seed_{seed}"
        # out_dir = f"videos/output_re_text2video-synthesis/seed_{seed}"
        # if len(prompt) < 35:
        #         name = prompt.split('.')[0]
        # else: name = prompt[:35]
        # if not os.path.exists(out_dir):
        #         os.makedirs(out_dir)
        # in_path= os.path.join(in_dir,f"{name}_seed{seed}.mp4")
        # out_path = os.path.join(out_dir,f"{name.replace(' ','_')}_seed{seed}.mp4")

        #text2video-zero
        in_dir = f"../Text2Video-Zero/output_8fps_16len/seed_{seed}"
        out_dir = f"videos/output_re_Text2Video-Zero/seed_{seed}"
        if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        in_path= os.path.join(in_dir,f"{prompt.replace(' ','_')}_seed{seed}.mp4")
        out_path = os.path.join(out_dir,f"{prompt.split('.')[0].replace(' ','_')}_seed{seed}.mp4")

        p_input = {
                'video_path': in_path,
                'text': prompt
            }
  
        output_video_path = video_to_video_pipe(p_input, output_video=out_path)[OutputKeys.OUTPUT_VIDEO]
        print(output_video_path)
      
   
    