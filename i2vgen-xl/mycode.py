from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import torch
seed = 42
generator = torch.Generator().manual_seed(seed)
li = list(open("data/test_list_for_i2vgen.txt"))
print(li)
images = []
texts = []
names = []
for i in li:
    im = i.split('|||')[0].split('# ')[-1]
    name = im.split('/')[-1].split('.')[0]
    te = i.split("|||")[1].strip()
    images.append(im)
    texts.append(te)
    names.append(name)
# step1 image2video
image_to_video_pipe = pipeline(task="image-to-video", model='damo/Image-to-Video', model_revision='v1.1.0', device='cuda:1')
for im,te,name in zip(images,texts,names):
    image_in = im
    i2v_output = f'output/output_1/i2v_output_{name}.mp4'
    output_video_path = image_to_video_pipe(image_in, output_video=i2v_output,generator=generator)[OutputKeys.OUTPUT_VIDEO]
    print(output_video_path)
    print(f"finsh image2video:{name}")

'''
    step2 video2video
'''

# video_to_video_pipe = pipeline(task="video-to-video", model='damo/Video-to-Video', model_revision='v1.1.0', device='cuda:1')
# for im,te,name in zip(images,texts,names):
#     image_in = im
#     i2v_output = f'output/output_1/i2v_output_{name}.mp4'  #former pipeline
#     print("start video2video with prompt...")
#     #step2 video2video   best do steps seperately with image2video pipeline, otherwise the GPU memory can't handle.
#     text_in = te +'.'
#     output_path = f'output/output_1/v2v_output_{name}.mp4'
#     p_input = {
#                 'video_path': i2v_output,
#                 'text': text_in
#             }
#     output_video_path = video_to_video_pipe(p_input, output_video=output_path,generator=generator)[OutputKeys.OUTPUT_VIDEO]
#     print(output_video_path)
#     print(f"finsh video2video:{name}")