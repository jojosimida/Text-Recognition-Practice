import torch
import torch.utils.data as Data
from torchvision.transforms import ToTensor, ToPILImage

import numpy as np
import copy
import json
import pickle
import math
import os
import re
import random
from PIL import Image


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

class getImage(object):

	def __init__(self, opt):
		self.opt = opt
		self.train = opt.train_root
		self.test = opt.test_root

		self.train_image_dirs = self.getImagefile(self.train)
		self.test_image_dirs = self.getImagefile(self.test)

	def getImagefile(self, path):
		names = os.listdir(path)
		names.remove('gt.txt')
		dirs = [os.path.join(path, n) for n in names]
		dirs.sort(key=natural_keys)
		return dirs

		
class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, opt):
        self.opt = opt
        self.imgH = opt.imgH
        self.imgW = opt.imgW
        self.keep_ratio_with_pad = opt.PAD

    def processImage(self, dirs):

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3   
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for path in dirs:
                image = Image.open(path).convert('RGB')
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            image_tensors = []
            transform = ResizeNormalize((self.imgW, self.imgH))
            for path in dirs:
                image = Image.open(path).convert('RGB')
                image_tensors.append(transform(image))
            # image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors


class ProcessGroundTruth(object):
	
	def __init__(self, opt):
		self.opt = opt
		self.train = opt.train_root
		self.test = opt.test_root
		self.train_gt = self.train+'gt.txt'
		self.test_gt = self.test+'gt.txt'

		self.train_gt = self.processGT(self.train_gt)
		self.test_gt = self.processGT(self.test_gt)
		self.all_char = self.get_all_char(self.train_gt)
		self.convert2label()
		self.train_text_tensors = self.encodeGT(self.train_gt)
		self.test_text_tensors = self.encodeGT(self.test_gt)

	def processGT(self, gt):
		content = readfile(gt)
		labels = splitstring(content)
		return labels

	def get_all_char(self, labels):
		return list(set(''.join(labels)))


	def convert2label(self):
		list_token = ['<SOS>', '<EOS>', '<UNKNOWN>']
		self.character = list_token + self.all_char
		self.char2idx = {}
		self.idx2char = {}
		for i, char in enumerate(self.character):
			self.char2idx[char] = i
			self.idx2char[i] = char

		self.charsize = len(self.char2idx)

	def encodeGT(self, text):
		# padded with <EOS> token when the word is end.
		text_tensor = torch.LongTensor(len(text), self.opt.max_length).fill_(1)
		for i, t in enumerate(text):
			t = list(t)
			t.append('<EOS>')
			t = [self.char2idx[char] if char in self.char2idx else 2 for char in t]
			text_tensor[i][:len(t)] = torch.LongTensor(t)  
		return text_tensor


class Loader(Data.DataLoader):

	def __init__(self, opt, image, label, shuffle=False, device=torch.device('cpu')):
		self.batch_size = opt.batch_size
		data = [image, label]
		torch_dataset = Data.TensorDataset(*(x.to(device) for x in data))

		super().__init__(
			dataset=torch_dataset,
			batch_size=self.batch_size,
			shuffle=shuffle,
			num_workers=0,
			drop_last=False
			)


def readfile(data):
	with open(data, "r", encoding="utf-8") as f:
		content = f.read().splitlines()
	return content

def splitstring(content):
	labels = []
	for c in content:
		labels.append(c.split()[-1][1:-1])
	return labels


