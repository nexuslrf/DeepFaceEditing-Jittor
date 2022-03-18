import torch
import numpy as np
from torchvision import transforms
from PIL import Image

import networks_torch as networks
from argparse import ArgumentParser

img_size = 512
transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size = img_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def read_img(path):
    img = Image.open(path).convert('RGB')
    img = transform_image(img)
    img = img.unsqueeze(0)
    return img

def save_img(image, path):
    image = image.squeeze(0).detach().cpu().numpy()
    image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--geo", type=str, default = "./images/17115_sketch.png", help = "the path of geometry image")
    parser.add_argument("--appear", type=str, default = "./images/69451.png", help = "the path of appearance image")
    parser.add_argument("--output", type=str, default = "./results/sketch_result.png", help = "the path of output image")
    parser.add_argument("--cuda", type=int, default = 1, help = "use cuda or cpu: 0 , cpu; 1 , gpu")
    parser.add_argument("--geo_type", type=str, default="sketch", help = "extract geometry from image or sketch: sketch / image")
    parser.add_argument("--gen_sketch", action='store_true', help = "with --gen_sketch, extract sketch from real image")
    args = parser.parse_args()

    device = torch.device('cuda') if args.cuda else torch.device('cpu')

    if args.gen_sketch:
        sketch_netG = networks.GlobalGenerator(input_nc = 3, output_nc = 3, 
                                        ngf = 32, n_downsampling = 4, n_blocks = 9)
        ckpt = torch.load("./checkpoints/sketch_generator.pth")
        sketch_netG.load_state_dict(ckpt)
        sketch_netG.to(device)
        geo_img = read_img(args.geo)
        with torch.no_grad():
            sketch = sketch_netG(geo_img.to(device))
            save_img(sketch, args.output)
    else:
        geo_img = read_img(args.geo)
        appear_img = read_img(args.appear)
        model = networks.Combine_Model()
        model.initialize()
        geo_type = args.geo_type
        image_swap = model.inference(geo_img, appear_img, geo_type)
        save_img(image_swap, args.output)

"python test_model.py --geo ./images/sketch_gen.png --appear ./images/appearance.png --output ./results/sketch_result.png --geo_type sketch"
"python test_model.py --geo ./images/geometry.png --appear ./images/appearance.png --output ./results/image_result.png --geo_type image"
"python test_model.py --geo ./images/geometry.png --output ./results/sketch_gen.png --gen_sketch"

