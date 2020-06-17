import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision

import time
import numpy as np
import copy
from tqdm import tqdm
import pickle
from functools import reduce
from evaluation import *


class TextRecognition(nn.Module):

	def __init__(self, opt, groundtruth):
		super().__init__()
		self.opt = opt
		self.gt = groundtruth
		charsize = groundtruth.charsize
		self.convdrop = opt.conv_dropout

		self.channel_dim = [3, 32, 64, 128, 256, 512]
		self.pooling = [(2,2), (2,2), (2,1), (2,1), (2,1)]
		## get the width after convnet
		div = [s[1]  for s in self.pooling if s[1]!=1]
		d = reduce(lambda x,y: x*y, div)
		self.conv_w = self.opt.imgW//d    

# ========================================================================

		# self.conv_blocks = [self.conv_block(in_f, out_f, pooling) \
		# for in_f, out_f, pooling in zip(self.channel_dim, self.channel_dim[1:], 
		# 	self.pooling)]		
		# self.convnet = nn.Sequential(*self.conv_blocks)
		# for block in self.conv_blocks:
		# 	for layer in block:
		# 		if isinstance(layer, nn.Conv2d):
		# 			nn.init.xavier_normal_(layer.weight)
		# 			nn.init.normal_(layer.bias, 0)


		self.conv_blocks = [ResidualBlock(in_f, out_f, pooling, self.convdrop) \
		for in_f, out_f, pooling in zip(self.channel_dim, self.channel_dim[1:], 
			self.pooling)]	
		self.convnet = nn.Sequential(*self.conv_blocks)


		self._2maxlen = nn.Linear(self.conv_w, opt.max_length)
		self.init_linear(self._2maxlen)
# ========================================================================


		# encoder_layer  = nn.TransformerEncoderLayer(self.channel_dim[-1], opt.nhead, 
		# 					dim_feedforward=self.channel_dim[-1]*2, dropout=0.2)
		# self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
		# 					num_layers=opt.encoder_layer)


		self.BiLSTM = nn.LSTM(self.channel_dim[-1], self.channel_dim[-1]//2, num_layers=opt.bilstm_n, 
                              bidirectional=True, batch_first=True, dropout=opt.bilstm_dropout) 	
		for param in self.BiLSTM.parameters():
			nn.init.normal_(param.data, mean=0, std=np.sqrt(1.0 / (self.channel_dim[-1])))


		# self.LSTM = nn.LSTM(self.channel_dim[-1], self.channel_dim[-1], num_layers=1, 
		#                             bidirectional=False, batch_first=True)
		# for param in self.LSTM.parameters():
		# 	nn.init.normal_(param.data, mean=0, std=np.sqrt(1.0 / (self.channel_dim[-1])))

# ========================================================
		# self.resnet50 = torchvision.models.resnet50(pretrained=True)
		# for param in self.resnet50.parameters():
		# 	param.requires_grad = False
		# self.resnet50.avgpool = nn.AdaptiveAvgPool2d((1,opt.max_length))
		# self.resnet50.fc = nn.Identity()
		# self.final = nn.Linear(2048, charsize)
# ========================================================


		self.fc = nn.Linear(self.channel_dim[-1], charsize)
		self.dropout = nn.Dropout(0.15) 
		self.softmax = nn.LogSoftmax(dim=-1)


	def conv_block(self, in_f, out_f, pooling):
		return nn.Sequential(
		nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
		nn.BatchNorm2d(out_f),
		nn.ReLU(),
		nn.Dropout(self.opt.conv_dropout),
		nn.Conv2d(out_f, out_f, kernel_size=3, padding=1),	
		nn.BatchNorm2d(out_f),
		nn.ReLU(),
		nn.Dropout(self.opt.conv_dropout),
		nn.MaxPool2d(pooling),		
		)

	def init_linear(self, m):
		nn.init.xavier_normal_(m.weight)
		nn.init.normal_(m.bias)


	def forward(self, image):

		image = self.convnet(image)                   ## B x 512 x 1 x conv_w
		image = F.relu(self._2maxlen(image))          ## B x 512 x 1 x max_length
		image = image.permute(0,3,1,2).squeeze(-1)    ## B x max_length x 512

#=======transformer========		
		# image = self.transformer_encoder(image)  
#==========================

		image = self.BiLSTM(image)[0]                 ## B x max_length x 512
		# image = self.LSTM(image)[0]  

		text = self.fc(image)
		text = self.softmax(text)                     ## B x max_length x charsize  

		return text
		


	def fit(self, train_loader, valid_loader):

		nllloss = nn.NLLLoss()
		optimizer = optim.AdamW(self.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)

		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.scheduler_step, 
                                              gamma=self.opt.scheduler_gamma)


		for epoch in tqdm(range(self.opt.epoch)):
			for images, texts in train_loader:
				self.train()
				output = self.forward(images)
				loss = nllloss(output.transpose(1, 2), texts)
				loss.backward()
				# torch.nn.utils.clip_grad_value_(self.parameters(), 0.5)

				# update parameters
				optimizer.step()               
				optimizer.zero_grad()
			scheduler.step()

			for valid_images, valid_texts in valid_loader:
				self.eval()
				valid_output = self.forward(valid_images)
				valid_loss = nllloss(valid_output.transpose(1, 2), valid_texts)


			print("epoch: %d | loss %.4f" % (epoch+1, loss))
			print("      %s  | val loss %.4f" % (" "*len(str(epoch+1)), valid_loss))

			if epoch%20==19:
				train_acc = self.score(train_loader)
				valid_acc = self.score(valid_loader)
				print('Train: ', train_acc)
				print('Validation: ', valid_acc)

		return self



	def score(self, loader):
		evaluation = Evaluation(self, loader)
		acc = evaluation.evaluate()
		return acc


	def predict(self, ):
		pass



class ResidualBlock(nn.Module):
	def __init__(self, in_f, out_f, pooling, dropout):
		super().__init__()
		self.conv1 = nn.Conv2d(in_f, out_f, kernel_size=3, padding=1)
		self.BN1 = nn.BatchNorm2d(out_f)
		self.dropout1 = nn.Dropout(dropout)
		self.conv2 = nn.Conv2d(out_f, out_f, kernel_size=3, padding=1)
		self.BN2 = nn.BatchNorm2d(out_f)
		self.dropout2 = nn.Dropout(dropout)
		self.maxpooling = nn.MaxPool2d(pooling)

		nn.init.xavier_normal_(self.conv1.weight)
		nn.init.normal_(self.conv1.bias)
		nn.init.xavier_normal_(self.conv2.weight)
		nn.init.normal_(self.conv2.bias)


	def forward(self,x):
		out = self.dropout1(F.relu(self.BN1(self.conv1(x))))
		residual = out
		out = self.dropout2(F.relu(self.BN2(self.conv2(out))))      
		out += residual
		return self.maxpooling(out)
