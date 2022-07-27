#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image

from logger import setup_logger
from model import BiSeNet


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path="vis_results/parsing_map_on_im.jpg"):
    # Colors for all 20 parts
    part_colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 0, 85],
        [255, 0, 170],
        [0, 255, 0],
        [85, 255, 0],
        [170, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [0, 85, 255],
        [0, 170, 255],
        [255, 255, 0],
        [255, 255, 85],
        [255, 255, 170],
        [255, 0, 255],
        [255, 85, 255],
        [255, 170, 255],
        [0, 255, 255],
        [85, 255, 255],
        [170, 255, 255],
    ]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] + ".png", vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im


def evaluate(res_path, images_path, ckp_path):

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(ckp_path))
    net.eval()

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    with torch.no_grad():
        for image_path in os.listdir(images_path):
            img = Image.open(osp.join(images_path, image_path)).convert("RGB")
            image = img.resize((448, 448), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            print("out shape:", out.shape)
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            print("parsing shape:", parsing.shape)
            # print(parsing)
            print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(res_path, image_path))


def deform_input(inp, deformation):
    _, h_old, w_old, _ = deformation.shape
    _, _, h, w = inp.shape
    if h_old != h or w_old != w:
        deformation = deformation.permute(0, 3, 1, 2)
        deformation = F.interpolate(deformation, size=(h, w), mode="bilinear")
        deformation = deformation.permute(0, 2, 3, 1)
    return F.grid_sample(inp, deformation, align_corners=False)


if __name__ == "__main__":
    img_path = "talkinghead-val/"
    res_path = "res"
    ckp_path = "ckp/79999_iter.pth"
    evaluate(res_path, img_path, ckp_path)
    # out = torch.randn((1, 19, 512, 512))
    # deform = torch.randn((1, 256, 256, 2))
    # out = deform_input(out, deform)
    # print(out.shape)
