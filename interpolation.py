#!/usr/bin/env python

# for file operation
import os
import argparse
import re
import shutil

# for image I/O
import cv2
import PIL.Image

# for EMA-VFI
import torch
import numpy as np

import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

##########################################################
# parse arguments
##########################################################

parser = argparse.ArgumentParser()

parser.add_argument('--src', type=str, required=True)
parser.add_argument('--dst', type=str, required=True)
parser.add_argument('--dst_filename', type=str, default='%08d.png')
# ex. specified 2, 30fps to 60fps
parser.add_argument('--factor', '--n', type=int, default=2)
parser.add_argument('--model', default='ours_t', type=str)

args = parser.parse_args()

##########################################################
# EMA-VFI model setup
##########################################################

pyDirPath = os.path.dirname(__file__)
modelPath = os.path.join(pyDirPath, 'model', args.model + '.pkl')
assert not os.path.exists(modelPath), 'Model not exists!'

TTA = True
if args.model == 'ours_small_t':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = args.model
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
model = Model(-1)
model.load_model()
model.eval()
model.device()

##########################################################
# Interpolation
##########################################################

def doInterpolate(srcFirst, srcSecond, dstFirstNum):
    I0 = cv2.imread(srcFirst)
    I2 = cv2.imread(srcSecond)

    I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

    padder = InputPadder(I0_.shape, divisor=32)
    I0_, I2_ = padder.pad(I0_, I2_)

    preds = model.multi_inference(I0_, I2_, TTA=TTA, time_list=[(i+1)*(1./args.factor) for i in range(args.factor - 1)], fast_TTA=TTA)

    for i, pred in enumerate(preds):
        dstPath = os.path.join(args.dst, f'{args.dst_filename}' % (dstFirstNum + i))
        npyEstimate = (padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)[:, :, ::-1]

        print('[EMA-VFI] interpolation : ' + dstPath)
        PIL.Image.fromarray(npyEstimate).save(dstPath)

if not os.path.exists(args.dst):
    os.mkdir(args.dst)

imgFileRegex = re.compile('^(\d+)\.(jpg|png|jpeg|bmp)$')
files = [f for f in os.listdir(args.src) if os.path.isfile(os.path.join(args.src, f))]
imgFiles = [f for f in files if imgFileRegex.match(f)]
imgFiles.sort(key=lambda x: int(imgFileRegex.match(x).group(1)))

outIndex = 1
for i, imgFile in enumerate(imgFiles):
    currSrcPath = os.path.join(args.src, imgFile)

    cpDstPath = os.path.join(args.dst, f'{args.dst_filename}' % outIndex)

    shutil.copyfile(currSrcPath, cpDstPath)
    print('[EMA-VFI] copy          : ' + cpDstPath)

    if i + 1 < len(imgFiles):
        nextSrcPath = os.path.join(args.src, imgFiles[i + 1])
        doInterpolate(currSrcPath, nextSrcPath, outIndex + 1)

    outIndex += args.factor

print('Completed')
