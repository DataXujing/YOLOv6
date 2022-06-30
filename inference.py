#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import sys
import os.path as osp

import numpy as np
import cv2
from PIL import ImageFont
import torch
import math

from tqdm import tqdm


from yolov6.utils.events import LOGGER, load_yaml

from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox  #latterbox
from yolov6.utils.nms import non_max_suppression


# init
img_size = 640
half = False
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
agnostic_nms = True


img_paths = [os.path.join("./test_img/",file) for file in os.listdir("./test_img")]
save_dir = "./res"

weights = "./runs/train/exp/weights/last_ckpt.pt"
yaml = "./data/score.yaml"

save_txt = True
save_img = True

cuda =  torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

model = DetectBackend(weights, device=device)
stride = model.stride
print("==================stride:",stride)  #32
class_names = load_yaml(yaml)['names']
print("==================class_name:",class_names)

# Half precision
if half & (device.type != 'cpu'):
    model.model.half()
else:
    model.model.float()
    half = False


# some func
def precess_image(path, img_size, stride, half):
    '''Process image before image inference.'''
    try:
        img_src = cv2.imread(path)
        assert img_src is not None, f'Invalid image: {path}'
    except Exception as e:
        LOGGER.Warning(e)
    image = letterbox(img_src, img_size, stride=stride)[0]

    # Convert
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    return image, img_src

def rescale(ori_shape, boxes, target_shape):
    '''Rescale the output to the original image shape'''
    ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
    padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio

    boxes[:, 0].clamp_(0, target_shape[1])  # x1
    boxes[:, 1].clamp_(0, target_shape[0])  # y1
    boxes[:, 2].clamp_(0, target_shape[1])  # x2
    boxes[:, 3].clamp_(0, target_shape[0])  # y2

    return boxes

def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    # Add one xyxy box to image with label
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)

def font_check(font='./yolov6/utils/Arial.ttf', size=10):
    # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
    assert osp.exists(font), f'font path not exists: {font}'
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except Exception as e:  # download if missing
        return ImageFont.truetype(str(font), size)

def box_convert(x):
    # Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def generate_colors(i, bgr=False):
    hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
            '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
    palette = []
    for iter in hex:
        h = '#' + iter
        palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
    num = len(palette)
    color = palette[int(i) % num]
    return (color[2], color[1], color[0]) if bgr else color



# 开始识别文件夹中的文件
for img_path in tqdm(img_paths):
    img, img_src = precess_image(img_path, img_size, stride, half)
    img = img.to(device)
    if len(img.shape) == 3:
        img = img[None]
        # expand for batch dim
    pred_results = model(img)
    classes = None # not filter class
    hide_labels = False
    hide_conf = False
    det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

    save_path = osp.join(save_dir, osp.basename(img_path))  # im.jpg
    txt_path = osp.join(save_dir, 'labels', osp.basename(img_path).split('.')[0])

    gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    img_ori = img_src

    # check image and font
    assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
    font_check()

    if len(det):
        det[:, :4] = rescale(img.shape[2:], det[:, :4], img_src.shape).round()

        for *xyxy, conf, cls in reversed(det):
            if save_txt:  # Write to file
                xywh = (box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf)
                with open(txt_path + '.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if save_img:
                class_num = int(cls)  # integer class
                label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')

                plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=generate_colors(class_num, True))

        img_src = np.asarray(img_ori)

        # Save results (image with detections)
        if save_img:
            cv2.imwrite(save_path, img_src)














