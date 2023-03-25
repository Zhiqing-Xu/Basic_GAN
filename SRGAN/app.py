#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#--------------------------------------------------#
import os 
import sys
import os.path
from sys import platform
from pathlib import Path
#--------------------------------------------------#
if __name__ == "__main__":
    print("="*80)
    if os.name == 'nt' or platform == 'win32':
        print("Running on Windows")
        if 'ptvsd' in sys.modules:
            print("Running in Visual Studio")
#--------------------------------------------------#
    if os.name != 'nt' and platform != 'win32':
        print("Not Running on Windows")
#--------------------------------------------------#
    if "__file__" in globals().keys():
        try:
            os.chdir(os.path.dirname(__file__))
            print('CurrentDir: ', os.getcwd())
        except:
            print("Problems with navigating to the file dir.")
    else:
        print("Running in python jupyter notebook.")
        try:
            if not 'workbookDir' in globals():
                workbookDir = os.getcwd()
                print('workbookDir: ' + workbookDir)
                os.chdir(workbookDir)
        except:
            print("Problems with navigating to the workbook dir.")
#--------------------------------------------------#

import gradio as gr
import torch
import cv2

import imgproc
from imgproc import image_to_tensor
from inference import choice_device, build_model
from utils import load_state_dict

model = "srresnet_x4"

device = choice_device("cpu")

# Initialize the model
sr_model = build_model(model, device)
print(f"Build {model} model successfully.")

# Load model weights
sr_model = load_state_dict(sr_model, "./weights/SRGAN_x4-ImageNet-8c4a7569.pth.tar")
print(f"Load `{model}` model weights successfully.")

# Start the verification mode of the model.
sr_model.eval()

def downscale(image):
    (width, height, colors) = image.shape

    new_height = int(60 * width / height)

    return cv2.resize(image, (60, new_height), interpolation=cv2.INTER_AREA)


def preprocess(image):
    image = image / 255.0

    # Convert image data to pytorch format data
    tensor = image_to_tensor(image, False, False).unsqueeze_(0)

    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.to(device="cpu", memory_format=torch.channels_last, non_blocking=True)

    return tensor

def processHighRes(image):
    if image is None:
        raise gr.Error("Please enter an image")
    downscaled = downscale(image)
    lr_tensor = preprocess(downscaled)

    # Use the model to generate super-resolved images
    with torch.no_grad():
        sr_tensor = sr_model(lr_tensor)

    # Save image
    sr_image = imgproc.tensor_to_image(sr_tensor, False, False)

    return [downscaled, sr_image]

def processLowRes(image):
    if image is None:
        raise gr.Error("Please enter an image")

    (width, height, colors) = image.shape

    if width > 400 or height > 400:
        raise gr.Error("Image is too big")

    lr_tensor = preprocess(image)

    # Use the model to generate super-resolved images
    with torch.no_grad():
        sr_tensor = sr_model(lr_tensor)

    # Save image
    sr_image = imgproc.tensor_to_image(sr_tensor, False, False)

    return sr_image

description = """<p style='text-align: center'> <a href='https://arxiv.org/abs/1609.04802' target='_blank'>Paper</a> | <a href=https://github.com/Lornatang/SRGAN-PyTorch target='_blank'>GitHub</a></p>"""
about = "<p style='text-align: center'>Made for the 2022-2023 Grenoble-INP Phelma Image analysis course by Thibaud CHERUY, Clément DEBUY & Yassine EL KHANOUSSI.</p>"

with gr.Blocks() as demo:
    gr.Markdown("# **<p align='center'>SRGAN: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network</p>**")
    gr.Markdown(description)

    with gr.Tab("From high res"):
        high_res_input = gr.Image(label="High-res source image", show_label=True)
        with gr.Row():
            low_res_output = gr.Image(label="Low-res image")
            srgan_output = gr.Image(label="SRGAN Output")
        high_res_button = gr.Button("Process")

    with gr.Tab("From low res"):
        low_res_input = gr.Image(label="Low-res source image", show_label=True)
        srgan_upscale = gr.Image(label="SRGAN Output")
        low_res_button = gr.Button("Process")

    gr.Examples(
        examples=["examples/bird.png", "examples/butterfly.png", "examples/comic.png", "examples/gray.png",
                  "examples/man.png"],
        inputs=[high_res_input],
        outputs=[low_res_output, srgan_output],
        fn=processHighRes
    )

    high_res_button.click(processHighRes, inputs=[high_res_input], outputs=[low_res_output, srgan_output])
    low_res_button.click(processLowRes, inputs=[low_res_input], outputs=[srgan_upscale])

    gr.Markdown(about)

demo.launch(share=True)
