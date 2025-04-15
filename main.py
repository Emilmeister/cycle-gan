import albumentations as A
import torch
import torch.nn as nn
import torch.nn.parallel
from albumentations.pytorch import ToTensorV2
import gradio as gr
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      4,
                      stride,
                      padding=1,
                      padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, down=True, use_act=True, **kwargs):
        super().__init__()
        if down:
            conv = nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=3,
                             stride=stride,
                             padding=1,
                             padding_mode='reflect',
                             **kwargs)
        else:
            conv = nn.ConvTranspose2d(in_channels,
                                      out_channels,
                                      kernel_size=3,
                                      stride=stride,
                                      padding=1,
                                      **kwargs)

        norm = nn.InstanceNorm2d(out_channels)
        if use_act:
            act = nn.ReLU(inplace=True)
        else:
            act = nn.Identity()

        self.conv = nn.Sequential(conv, norm, act)

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res = nn.Sequential(
            ConvBlock(channels,
                      channels,
                      stride=1),

            ConvBlock(channels,
                      channels,
                      stride=1,
                      use_act=False))

    def forward(self, x):
        return x + self.res(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_resudials=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=7,
                      stride=1,
                      padding=3),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True))

        self.down = nn.Sequential(
            ConvBlock(out_channels,
                      out_channels * 2),
            ConvBlock(out_channels * 2,
                      out_channels * 4)
        )

        self.res = nn.Sequential(*[ResidualBlock(out_channels * 4) for _ in range(num_resudials)])

        self.up = nn.Sequential(
            ConvBlock(out_channels * 4,
                      out_channels * 2,
                      down=False,
                      output_padding=1),
            ConvBlock(out_channels * 2,
                      out_channels,
                      down=False,
                      output_padding=1))
        self.last = nn.Conv2d(out_channels,
                              in_channels,
                              kernel_size=7,
                              stride=1,
                              padding=3,
                              padding_mode="reflect"
                              )
        self.lambda1 = nn.Parameter(torch.tensor(1.0))
        self.lambda2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x1 = self.initial(x)
        x2 = self.down(x1)
        x3 = self.res(x2)
        x4 = self.up(x3) + self.lambda1 * x1
        x5 = self.last(x4) + self.lambda2 * x

        return torch.tanh(x5)


generator = Generator(out_channels=96)
generator.load_state_dict(torch.load("model/gen_mtp", weights_only=True, map_location=torch.device('cpu')))
generator.eval()

SIZE = 256

model_path = 'model/RealESRGAN_x4plus.pth'
state_dict = torch.load(model_path)['params_ema']

model = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64,
    num_block=23, num_grow_ch=32, scale=4
)
model.load_state_dict(state_dict, strict=True)

# Initialize upsampler with tiling
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=400,  # Split image into tiles for processing
    tile_pad=10,  # Padding for tiles
    pre_pad=0,
    half=True if torch.cuda.is_available() else False  # Disable FP16 on CPU
)


def apply_filter(input_img):
    pil_img = Image.fromarray(input_img)
    width, height = pil_img.size
    size = min(width, height)
    img = np.array(pil_img.crop((1, 1, size, size)).resize((SIZE, SIZE)).convert("RGB"))
    prepare = A.Compose([A.Resize(height=SIZE, width=SIZE),
                         A.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5),
                                     max_pixel_value=255.0),
                         ToTensorV2()])
    img = prepare(image=img)['image']
    img = generator.forward(img)
    img = torch.ceil((img + 1) / 2 * 255)
    scaled_img, _ = upsampler.enhance(img.transpose(0, 2).transpose(0, 1).detach().numpy(), outscale=4)
    return scaled_img


with gr.Blocks() as demo:
    gr.Markdown("# 🖼️ Картина в стиле Васильева")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Входное изображение")
            filter_btn = gr.Button("Применить фильтр")
        with gr.Column():
            output_image = gr.Image(label="В стиле Васильева")

    filter_btn.click(apply_filter, inputs=input_image, outputs=output_image)

    gr.Examples(
        examples=[
            "https://i.pinimg.com/originals/8f/89/a3/8f89a3eeb9990cbe065a2f98dbb52ee3.jpg",
            "https://avatars.mds.yandex.net/get-mpic/5341376/2a0000018b438b7ebf3f1db92a5dd4788766/orig",
            "https://cdn1.ozone.ru/multimedia/1025361279.jpg"
        ],
        inputs=input_image
    )

demo.launch()
