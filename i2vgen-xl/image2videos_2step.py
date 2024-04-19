# !usr/bin/env python3
# -*- coding:UTF-8 -*-
# @Author: Xandra
# @File:image2videos_2step.py
# @Time:2024-04-16
'''
    This file produce the videos of 4s of the input images.
    input: image dir; text prompt
    two steps: 
        step1: image2video pipeline. You can finish here if the videos is satisfying.
        step2: video2video pipeline to get a high quality videos.
 
    reference: https://modelscope.cn/models/iic/i2vgen-xl/summary
               https://huggingface.co/ali-vilab/i2vgen-xl
               demo: https://modelscope.cn/studios/iic/I2VGen-XL/summary
                     https://huggingface.co/spaces/modelscope/I2VGen-XL
'''
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import os

def i2v(im_dir,i2v_out_dir):
    '''
    step1 image2video

    Args:
        im_dir: input dir 
        i2v_out_dir: output dir
    Returns:
        i2v_outputs: path list of all generated videos

    '''

    if not os.path.exists(i2v_out_dir):
        os.makedirs(i2v_out_dir)
    paths = os.listdir(im_dir) # images in from the dir
    image_to_video_pipe = pipeline(task="image-to-video", model='damo/Image-to-Video', model_revision='v1.1.0', device='cuda:0')
    i2v_outputs = []
    for path in paths:
        image_in = os.path.join(im_dir,path)
        i2v_output = os.path.join(i2v_out_dir,f"{path.split('/')[-1].split('.')[0]}.mp4")
        output_video_path = image_to_video_pipe(image_in, output_video=i2v_output)[OutputKeys.OUTPUT_VIDEO]
        i2v_outputs.append(output_video_path)
        print(output_video_path)
    return i2v_outputs

def v2v(v2v_out_dir,i2v_outputs,text_in):
    '''
    step2: video2video pipeline to get a high quality videos.

    Args:
        v2v_out_dir: output dir
        i2v_outputs: path list of input videos
        text_in: prompt
    Returns:
        v2v_outputs: path list of all generated videos

    '''
    if not os.path.exists(v2v_out_dir):
            os.makedirs(v2v_out_dir)
    video_to_video_pipe = pipeline(task="video-to-video", model='damo/Video-to-Video', model_revision='v1.1.0', device='cuda:0')
    print("start video2video with prompt...")
    v2v_outputs = []
    for i in i2v_outputs:
        v2v_output = os.path.join(v2v_out_dir,i.split('/')[-1]) 
        p_input = {
                    'video_path': i,
                    'text': text_in
                }
        output_video_path = video_to_video_pipe(p_input, output_video=v2v_output)[OutputKeys.OUTPUT_VIDEO]
        print(output_video_path)
        v2v_outputs.append(v2v_output)
    return v2v_outputs

if __name__ == '__main__':
    # i2v
    im_dir = "i2v_pic/cat"  #image dir with input images
    i2v_out_dir ='videos/i2v_outvideos_0/cat'#save dir after the image2video pipeline
    i2v_outputs= i2v(im_dir,i2v_out_dir)
    # v2v
    text_in = 'A dog.' # the prompt
    v2v_out_dir ='videos/v2v_outvideos_0/dog' #save dir after video2video pipeline
    v2v(v2v_out_dir, i2v_outputs,text_in)
    

    # v2v with self-defined video path list
    # im_dir = "i2v_pic/dog"  #image dir with input images
    # i2v_out_dir ='videos/i2v_outvideos_0/dog'#save dir after the image2video pipeline
    # paths = os.listdir(im_dir) # images in from the dir
    # i2v_outputs = [os.path.join(i2v_out_dir,f"{path.split('/')[-1].split('.')[0]}.mp4") for path in paths]
    # text_in = 'A dog.' # the prompt
    # v2v_out_dir ='videos/v2v_outvideos_0/dog' #save dir after video2video pipeline
    # v2v(v2v_out_dir, i2v_outputs,text_in)