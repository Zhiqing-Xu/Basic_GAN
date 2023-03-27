import torch
from PIL import Image
import numpy as np
from realesrgan import RealESRGAN
import os
import gradio as gr
os.system("gdown https://drive.google.com/uc?id=1pG2S3sYvSaO0V0B8QPOl1RapPHpUGOaV -O RealESRGAN_x2.pth")
os.system("gdown https://drive.google.com/uc?id=1SGHdZAln4en65_NQeQY9UjchtkEF9f5F -O RealESRGAN_x4.pth")
os.system("gdown https://drive.google.com/uc?id=1mT9ewx86PSrc43b-ax47l1E2UzR7Ln4j -O RealESRGAN_x8.pth")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model2 = RealESRGAN(device, scale=2)
model2.load_weights('RealESRGAN_x2.pth')
model4 = RealESRGAN(device, scale=4)
model4.load_weights('RealESRGAN_x4.pth')
model8 = RealESRGAN(device, scale=8)
model8.load_weights('RealESRGAN_x8.pth')


def inference(image: Image, size: str) -> Image:
    if size == '2x':
        result = model2.predict(image.convert('RGB'))
    elif size == '4x':
        result = model4.predict(image.convert('RGB'))
    else:
        result = model8.predict(image.convert('RGB'))
    return result


title = "Face Real ESRGAN: 2x 4x 8x"
description = "This is an unofficial demo for Real-ESRGAN. Scales the resolution of a photo. This model shows better results on faces compared to the original version.<br>Telegram BOT: https://t.me/restoration_photo_bot"
article = "<div style='text-align: center;'>Twitter <a href='https://twitter.com/DoEvent' target='_blank'>Max Skobeev</a> | <a href='https://huggingface.co/sberbank-ai/Real-ESRGAN' target='_blank'>Model card</a>  <center><img src='https://visitor-badge.glitch.me/badge?page_id=max_skobeev_face_esrgan' alt='visitor badge'></center></div>"

gr.Interface(inference,

    [gr.inputs.Image(type="pil"), 
    
    gr.inputs.Radio(['2x', '4x', '8x'], 
    type="value", 
    default='2x', 
    label='Resolution model')], 
    gr.outputs.Image(type="pil", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=[['groot.jpeg', "2x"]],
    allow_flagging='never',
    theme="default",
    cache_examples=False,
    ).queue().launch()
    