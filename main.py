import numpy as np
import copy
import json
import pickle
import os
import argparse
from pprint import pprint
from sklearn.model_selection import train_test_split

import torch
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as Data
from torchvision.transforms import ToTensor, ToPILImage

from data_util import getImage, AlignCollate, ProcessGroundTruth, Loader
from models import TextRecognition



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', type=str, default='Challenge1_Training_Task3_Images_GT/')
    parser.add_argument('--test_root', type=str, default='Challenge1_Test_Task3_Images_GT/')
    parser.add_argument('--USE_CUDA', type=str2bool, default=True)
    parser.add_argument('--CUDA_device', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')

    ### Data processing ###
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--PAD', type=str2bool, default=True, help='whether to keep ratio then pad for image resize')
    parser.add_argument('--valid_ratio', type=float, default=0.05, help='The ratio of validation set in training set')
    parser.add_argument('--max_length', type=int, default=40, help='maximum-label-length')

    ### Model Architecture ###
    parser.add_argument('--epoch', type=int, default=600, help='The number of epoch')
    parser.add_argument('--bilstm_n', type=int, default=2, help='The number of bilstm layer')
    parser.add_argument('--bilstm_dim', type=int, default=256, help='The dimension of bilstm layer')
    parser.add_argument('--bilstm_dropout', type=float, default=0.4)
    parser.add_argument('--conv_dropout', type=float, default=0.25)
    
    parser.add_argument('--d_model', type=int, default=256, help='The dimension of transformer encoder layer')
    parser.add_argument('--encoder_layer', type=int, default=3, help='Number of transformer encoder layers')
    parser.add_argument('--nhead', type=int, default=4, help='The number of heads in the multiheadattention models')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='The dimension of the feedforward network model')
    
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate in AdamW')
    parser.add_argument('--weight_decay', type=float, default=2e-2)
    parser.add_argument('--scheduler_step', type=float, default=25)
    parser.add_argument('--scheduler_gamma', type=float, default=0.9)


    opt  = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.CUDA_device
    use_device = torch.device('cuda' if opt.USE_CUDA else 'cpu')


    _getImage = getImage(opt)
    _AlignCollate = AlignCollate(opt)
    train_image_tensors = _AlignCollate.processImage(_getImage.train_image_dirs)
    test_image_tensors = _AlignCollate.processImage(_getImage.test_image_dirs)

    _ProcessGroundTruth = ProcessGroundTruth(opt)
    train_text_tensors =  _ProcessGroundTruth.train_text_tensors
    test_text_tensors = _ProcessGroundTruth.test_text_tensors

    
    split = \
    train_test_split(train_image_tensors.numpy(), train_text_tensors.numpy(), test_size=opt.valid_ratio)
    X_train, X_valid, y_train, y_valid = map(torch.from_numpy, split)


    train_loader = Loader(opt, X_train, y_train, shuffle=True, device=use_device)
    valid_loader = Loader(opt, X_valid, y_valid, shuffle=False, device=use_device)


    model = TextRecognition(opt, _ProcessGroundTruth).to(use_device)
    model.fit(train_loader, valid_loader)

    test_loader = Loader(opt, test_image_tensors, test_text_tensors, shuffle=False, device=use_device)
    test_acc = model.score(test_loader)
    print('Test: ', test_acc)

