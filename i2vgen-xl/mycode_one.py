from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import torch
seed = 0
generator = torch.Generator().manual_seed(seed)

# step1 image2video
image_to_video_pipe = pipeline(task="image-to-video", model='damo/Image-to-Video', model_revision='v1.1.0', device='cuda:1')

image_in = 'BB1lG1SA.jpg'
i2v_output = 'BB1lG1SA.mp4'
output_video_path = image_to_video_pipe(image_in, output_video=i2v_output,generator=generator)[OutputKeys.OUTPUT_VIDEO]
print(output_video_path)
print(f"finsh image2video")

'''
    step2 video2video
'''

video_to_video_pipe = pipeline(task="video-to-video", model='damo/Video-to-Video', model_revision='v1.1.0', device='cuda:1')

#step2 video2video   best do steps seperately with image2video pipeline, otherwise the GPU memory can't handle.
# your prompt for that video
text_in = 'A girl with black hair sitting along the river, at night.'
output_path = 'v2v.mp4'
p_input = {
                'video_path': i2v_output,
                'text': text_in
            }
output_video_path = video_to_video_pipe(p_input, output_video=output_path,generator=generator)[OutputKeys.OUTPUT_VIDEO]
print(output_video_path)
print(f"finsh video2video")