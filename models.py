import torch
import os
import sys
import numpy as np
from torch import nn
from torch import distributions as dists
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.nn import functional as F
#import utils as uts

# using MNIST as an example
class VAEBase(nn.Module):
	def __init__(self,latent_dim,input_dim,device,**kwargs):
		super().__init__()
		self.latent_dim = latent_dim
		self.input_dim = input_dim
		self.device = device
		self.enc_stack = None
		# for Gaussian encoding only
		self.enc_mean = None
		self.enc_logvar = None

		self.dec_stack = None
		self.use_mse = False # can be specified explicitly

	def calc_loss(self,forward_out) -> dict:
		xre_logits = forward_out[0]
		x = forward_out[1]
		z_mean = forward_out[2]
		z_logvar = forward_out[3]
		z = forward_out[4]
		output_dim = np.prod(x.size()[1:])
		kl_loss = torch.mean(-0.5 * torch.sum( 1+ z_logvar - z_mean **2 - z_logvar.exp(), dim=1),dim=0)
		if self.use_mse:
			# for convenience, wrong name
			bce_loss = F.mse_loss(xre_logits.view(-1,output_dim),x.view(-1,output_dim),reduction='none')
			bce_loss = torch.mean(torch.sum(bce_loss,dim=1),dim=0)
		else:
			bce_loss = F.binary_cross_entropy_with_logits(xre_logits.view(-1,output_dim),x.view(-1,output_dim),reduction="none")
			bce_loss = torch.mean(torch.sum(bce_loss,dim=1),dim=0)
		return {
			"loss":bce_loss + kl_loss,
			"reconstruction":bce_loss,
			"kl_divergence":kl_loss,
			}

# standard MLP vanilla VAE
class VanillaVAE(VAEBase):
	def __init__(self,latent_dim,input_dim,inter_dim_list,device,**kwargs):
		super(VanillaVAE,self).__init__(latent_dim,input_dim,device)
		self.hid_dim_list = inter_dim_list
		self.enc_stack = self.build_encoder_stack()
		# gaussian encoding
		self.enc_mean = nn.Linear(inter_dim_list[-1],self.latent_dim)
		self.enc_logvar = nn.Linear(inter_dim_list[-1],self.latent_dim)
		self.dec_stack = self.build_decoder_stack()
		self.use_mse = kwargs.get("use_mse",False)

	def build_encoder_stack(self):
		# in order
		block_list = nn.ModuleList()
		if isinstance(self.input_dim,tuple):
			in_feat = np.prod(list(self.input_dim))
		else:
			in_feat = self.input_dim
		for idx, hid_dim in enumerate(self.hid_dim_list):
			block_list.append(
					nn.Sequential(
							nn.Linear(in_feat,hid_dim),
							nn.BatchNorm1d(hid_dim),
							nn.ReLU(inplace=True),
						)
					)
			in_feat = hid_dim
		return nn.Sequential(*block_list)

	def build_decoder_stack(self):
		block_list = nn.ModuleList()
		in_feat = self.latent_dim
		for idx, hid_dim in enumerate(np.flip(self.hid_dim_list)):
			block_list.append(
					nn.Sequential(
							nn.Linear(in_feat,hid_dim),
							nn.BatchNorm1d(hid_dim),
							nn.ReLU(inplace=True),
						)
				)
			in_feat = hid_dim
		if isinstance(self.input_dim,tuple):
			block_list.append(nn.Linear(hid_dim,np.prod(list(self.input_dim))))
		else:
			block_list.append(nn.Linear(hid_dim,self.input_dim)) # output as logits
		return nn.Sequential(*block_list)

	def forward(self,inputs):
		x = torch.flatten(inputs,start_dim=1)
		x = self.enc_stack(x)
		z_mean = self.enc_mean(x)
		z_logvar = self.enc_logvar(x)
		epsilon = torch.randn(size=z_mean.size()).to(self.device)
		z = z_mean + (0.5* z_logvar).exp() * epsilon
		xre_logits = self.dec_stack(z)
		return [xre_logits,inputs,z_mean,z_logvar,z]

	def decode(self,zin):
		xre_flat = self.dec_stack(zin)
		if not self.use_mse:
			xre_flat = xre_flat.sigmoid()
		tmp_shape = tuple([-1] + list(self.input_dim))
		xre_reshape = xre_flat.view(tmp_shape).detach()
		return [xre_reshape,zin.detach()]


# using MNIST as an example

class VAE(VAEBase):
	def __init__(self,latent_dim,input_dim,hidden_layer_channels,device,**kwargs):
		super(VAE,self).__init__(latent_dim,input_dim,device)
		#self.input_dim = input_dim # (1,28,28), channel, w,h
		self.hidden_layer_channels = hidden_layer_channels

		self.inter_dim = int(np.prod(list(input_dim)[1:])/16)
		# encoder blocks
		self.enc_stack = nn.Sequential(
				nn.Conv2d(input_dim[0],self.hidden_layer_channels,3,stride=2,padding=1), # padding number from trial
				nn.ReLU(),
				nn.Conv2d(self.hidden_layer_channels,self.hidden_layer_channels,3,stride=2,padding=1),
				nn.ReLU(),
			)
		self.enc_mean = nn.Linear(self.hidden_layer_channels*self.inter_dim,latent_dim)
		self.enc_logvar= nn.Linear(self.hidden_layer_channels*self.inter_dim,latent_dim)

		# decoder blocks
		self.dec_stack = nn.Sequential(
				nn.Linear(latent_dim,self.inter_dim*self.hidden_layer_channels),
				nn.ReLU(),
			)
		self.dec_cnn = nn.Sequential(
				nn.ConvTranspose2d(self.hidden_layer_channels,self.hidden_layer_channels,3,stride=2,padding=1,output_padding=1), # same padding number of input padding and output paddings
				nn.ReLU(),
				nn.ConvTranspose2d(self.hidden_layer_channels,input_dim[0],3,stride=2,padding=1,output_padding=1),
			)
		self.use_mse = kwargs.get("use_mse",False)

	def forward(self,inputs):
		x = self.enc_stack(inputs)
		x = x.view(-1,self.hidden_layer_channels*self.inter_dim)
		z_mean = self.enc_mean(x)
		z_logvar=self.enc_logvar(x)
		epsilon = torch.randn(size=z_mean.size()).to(self.device)
		z = z_mean + (0.5*z_logvar).exp() * epsilon

		# the kl loss is available here...
		x_re = self.dec_stack(z)
		tmp_list =list(self.input_dim)[1:]
		rec_shape = tuple([-1,self.hidden_layer_channels]+[int(item/4) for item in tmp_list])
		x_re = x_re.view(rec_shape)
		rec_logits = self.dec_cnn(x_re)

		return [rec_logits,inputs,z_mean,z_logvar,z] # reconstruction_logits, input, rep_mean, rep_logvar
	def decode(self,zin):
		zre_flat = self.dec_stack(zin)
		tmp_list = list(self.input_dim)[1:]
		rec_shape = tuple([-1,self.hidden_layer_channels]+[int(item/4) for item in tmp_list])
		zre_reshape = zre_flat.view(rec_shape)
		xre = self.dec_cnn(zre_reshape)
		if not self.use_mse:
			xre = xre.sigmoid()
		return [xre.detach(),zin.detach()]


class VBerAEBase(nn.Module):
	def __init__(self,latent_dim,input_dim,device,**kwargs):
		super().__init__()
		#self.training = True # training flag
		self.latent_dim = latent_dim
		self.input_dim = input_dim
		self.device = device
		self.enc_stack = None
		# for Bernoulli logits
		self.enc_logits = None
		self.dec_stack = None
		self.use_mse = False

	def calc_loss(self,forward_out,**kwargs) -> dict:
		xre_sigmoid = kwargs.get("sigmoid",False)
		xre_logits = forward_out[0] # this is the same
		if xre_sigmoid:
			xre_logits = xre_logits.sigmoid()

		x = forward_out[1] # this is the input, so the same
		z_logits = forward_out[2] # this is the Ber logits
		#z_soft = forward_out[3] # this is the soft-probabilities, for training only
		#z      = forward_out[4] # this is the hard random samples
		output_dim = np.prod(x.size()[1:])
		# Bernoulli modeling
		kl_loss = torch.mean(torch.sum( np.log(2)+z_logits * z_logits.sigmoid() - F.softplus(z_logits),dim=1),dim=0)
		# the reconstruction loss is the same
		if self.use_mse:
			rec_loss = F.mse_loss(xre_logits.view(-1,output_dim),x.view(-1,output_dim),reduction="none")
			rec_loss = torch.mean(torch.sum(rec_loss,dim=1),dim=0)
		else:
			rec_loss = F.binary_cross_entropy_with_logits(xre_logits.view(-1,output_dim),x.view(-1,output_dim),reduction='none')
			rec_loss = torch.mean(torch.sum(rec_loss,dim=1),dim=0)
		return {
			"loss":rec_loss + kl_loss,
			"reconstruction":rec_loss,
			"kl_divergence":kl_loss,
			}

# standard MLP vanilla VAE
class VanillaBerVAE(VBerAEBase):
	def __init__(self,latent_dim,input_dim,inter_dim_list,device,**kwargs):
		super(VanillaBerVAE,self).__init__(latent_dim,input_dim,device)
		self.hid_dim_list = inter_dim_list
		self.enc_stack = self.build_encoder_stack()

		self.enc_logits = nn.Linear(inter_dim_list[-1],self.latent_dim)
		self.dec_stack = self.build_decoder_stack()
		self.use_mse = kwargs.get("use_mse",False)

	def build_encoder_stack(self):
		# in order
		block_list = nn.ModuleList()
		if isinstance(self.input_dim,tuple):
			in_feat = np.prod(list(self.input_dim))
		else:
			in_feat = self.input_dim
		for idx, hid_dim in enumerate(self.hid_dim_list):
			block_list.append(
					nn.Sequential(
							nn.Linear(in_feat,hid_dim),
							nn.BatchNorm1d(hid_dim),
							nn.ReLU(inplace=True),
						)
					)
			in_feat = hid_dim
		return nn.Sequential(*block_list)

	def build_decoder_stack(self):
		block_list = nn.ModuleList()
		in_feat = self.latent_dim
		for idx, hid_dim in enumerate(np.flip(self.hid_dim_list)):
			block_list.append(
					nn.Sequential(
							nn.Linear(in_feat,hid_dim),
							nn.BatchNorm1d(hid_dim),
							nn.ReLU(inplace=True),
						)
				)
			in_feat = hid_dim
		if isinstance(self.input_dim,tuple):
			block_list.append(nn.Linear(hid_dim,np.prod(list(self.input_dim))))
		else:
			block_list.append(nn.Linear(hid_dim,self.input_dim)) # output as logits
		return nn.Sequential(*block_list)

	def forward(self,inputs):
		x = torch.flatten(inputs,start_dim=1)
		x = self.enc_stack(x)
		z_logits = self.enc_logits(x)
		epsilon = torch.rand(size=z_logits.size()).to(self.device)
		z_soft = torch.sigmoid(z_logits - epsilon.log()) # smooth surrogate
		z_samp = torch.less(epsilon.log(),z_logits).float()
		if self.training:
			xre_logits = self.dec_stack(z_soft)
		else:
			xre_logits = self.dec_stack(z_samp)
		return [xre_logits,inputs,z_logits,z_soft,z_samp]
	def decode(self,zin):
		xre_flat = self.dec_stack(zin)
		if not self.use_mse:
			xre_flat = xre_flat.sigmoid()
		tmp_shape = tuple([-1] + list(self.input_dim))
		xre_reshape = xre_flat.view(tmp_shape).detach()
		return [xre_reshape,zin.detach()]


# using MNIST as an example
# so only have two stacks of CNN
class VBerAE(VBerAEBase):
	def __init__(self,latent_dim,input_dim,hidden_layer_channels,device,**kwargs):
		super(VBerAE,self).__init__(latent_dim,input_dim,device)
		#self.input_dim = input_dim # (1,28,28), channel, w,h
		self.hidden_layer_channels = hidden_layer_channels

		self.inter_dim = int(np.prod(list(input_dim)[1:])/16)
		# encoder blocks
		self.enc_stack = nn.Sequential(
				nn.Conv2d(input_dim[0],self.hidden_layer_channels,3,stride=2,padding=1), # padding number from trial
				nn.ReLU(),
				nn.Conv2d(self.hidden_layer_channels,self.hidden_layer_channels,3,stride=2,padding=1),
				nn.ReLU(),
			)
		self.enc_logits = nn.Linear(self.hidden_layer_channels*self.inter_dim,latent_dim)

		# decoder blocks
		self.dec_stack = nn.Sequential(
				nn.Linear(latent_dim,self.inter_dim*self.hidden_layer_channels),
				nn.ReLU(),
			)
		self.dec_cnn = nn.Sequential(
				nn.ConvTranspose2d(self.hidden_layer_channels,self.hidden_layer_channels,3,stride=2,padding=1,output_padding=1), # same padding number of input padding and output paddings
				nn.ReLU(),
				nn.ConvTranspose2d(self.hidden_layer_channels,input_dim[0],3,stride=2,padding=1,output_padding=1),
			)
		self.use_mse = kwargs.get("use_mse",False)

	def forward(self,inputs):
		x = self.enc_stack(inputs)
		x = x.view(-1,self.hidden_layer_channels*self.inter_dim)
		z_logits = self.enc_logits(x)
		epsilon = torch.rand(size=z_logits.size()).to(self.device)
		#if self.training:
		z_soft = torch.sigmoid(z_logits-epsilon.log()) # smooth surrogate
		z_samp = torch.less(epsilon.log(),z_logits).float()

		# the kl loss is available here...
		if self.training:
			x_re = self.dec_stack(z_soft)
		else:
			x_re = self.dec_stack(z_samp)
		tmp_list =list(self.input_dim)[1:]
		rec_shape = tuple([-1,self.hidden_layer_channels]+[int(item/4) for item in tmp_list])
		x_re = x_re.view(rec_shape)
		rec_logits = self.dec_cnn(x_re)

		return [rec_logits,inputs,z_logits,z_soft,z_samp] # reconstruction_logits, input, ...
	def decode(self,zin):
		zre_flat = self.dec_stack(zin)
		tmp_list = list(self.input_dim)[1:]
		rec_shape = tuple([-1,self.hidden_layer_channels]+[int(item/4) for item in tmp_list])
		zre_reshape = zre_flat.view(rec_shape)
		xre = self.dec_cnn(zre_reshape)
		if not self.use_mse:
			xre = xre.sigmoid()
		return [xre.detach(),zin.detach()]

def residual_block(num_stack=2):
	stack_list = nn.ModuleList()
	for ii in range(num_stack):
		stack = nn.Sequential(
				nn.Conv2d(2,8,3,padding=1),
				nn.BatchNorm2d(8),
				nn.LeakyReLU(),
				nn.Conv2d(8,16,3,padding=1),
				nn.BatchNorm2d(16),
				nn.LeakyReLU(),
				nn.Conv2d(16,2,3,padding=1),
			)
		stack_list.append(stack)
	return stack_list

class VBerCsiNet(VBerAEBase):
	def __init__(self,latent_dim,input_dim,device,**kwargs):
		super(VBerCsiNet,self).__init__(latent_dim,input_dim,device)
		self.use_mse = kwargs.get("use_mse",False)
		# define some sub-class specific member
		self.img_height=32
		self.img_width = 32
		self.img_channels =2
		self.enc_stack = self.build_inproc_block()
		self.enc_logits = nn.Linear(2048,latent_dim)
		self.dec_resnet_list = residual_block(num_stack=2)
		self.dec_stack = nn.Sequential(
				nn.Linear(latent_dim,self.img_height*self.img_width*self.img_channels),
				nn.BatchNorm1d(self.img_height*self.img_width*self.img_channels),
				nn.LeakyReLU(),
			)
		self.dec_post_proc = nn.Sequential(
				nn.Conv2d(2,2,3,stride=1,padding=1),
			)
	def build_inproc_block(self):
		stack = nn.Sequential(
				nn.Conv2d(2,2,3,stride=1,padding=1),
				nn.BatchNorm2d(2),
				nn.LeakyReLU(),
			)
		return stack
	def resnet_pass(self,x):
		for layer in self.dec_resnet_list:
			shortcut = x
			x = layer(x)
			x = shortcut +x
			x = F.leaky_relu(x)
		return x
	def forward(self,inputs):
		x = self.enc_stack(inputs)
		x = x.flatten(start_dim=1)
		z_logits = self.enc_logits(x)
		# bernoulli sampling
		epsilon = torch.rand(size=z_logits.size()).to(self.device)
		z_soft = torch.sigmoid(z_logits-epsilon.log()) # smooth surrogate
		z_samp = torch.less(epsilon.log(),z_logits).float()
		if self.training:
			z = z_soft
		else:
			z = z_samp
		# decoder
		z = self.dec_stack(z) # to input shape
		z = z.view((-1,self.img_channels,self.img_height,self.img_width))
		# go through residual blocks
		z = self.resnet_pass(z)
		# post processing
		xre_logits = self.dec_post_proc(z)
		# output the logits, 
		return [xre_logits,inputs,z_logits,z_soft,z_samp]
	def decode(self,zin):
		z = self.dec_stack(zin)
		z = z.view((self.img_channels,self.img_height,self.img_width))
		z = self.resnet_pass(z)
		xre = self.dec_post_proc(z)
		if not self.use_mse:
			xre = xre.sigmoid()
		return [xre.detach(),zin.detach()]


# reproduce CsiNet results
class CsiNet(nn.Module):
	def __init__(self,latent_dim,input_dim,device,**kwargs):
		super().__init__()
		self.latent_dim = latent_dim
		self.input_dim = input_dim
		self.device = device
		self.use_mse = kwargs.get("use_mse",False)
		self.img_height=32
		self.img_width = 32
		self.img_channels =2
		self.enc_stack = self.build_inproc_block()
		self.enc_logits = nn.Linear(2048,latent_dim)
		self.dec_resnet_list = residual_block(num_stack=2)
		self.dec_stack = nn.Sequential(
				nn.Linear(latent_dim,self.img_height*self.img_width*self.img_channels),
				nn.BatchNorm1d(self.img_height*self.img_width*self.img_channels),
				nn.LeakyReLU(),
			)
		self.dec_post_proc = nn.Sequential(
				nn.Conv2d(2,2,3,stride=1,padding=1),
			)
	def build_inproc_block(self):
		stack = nn.Sequential(
				nn.Conv2d(2,2,3,stride=1,padding=1),
				nn.BatchNorm2d(2),
				nn.LeakyReLU(),
			)
		return stack
	def resnet_pass(self,x):
		for layer in self.dec_resnet_list:
			shortcut = x
			x = layer(x)
			x = shortcut +x
			x = F.leaky_relu(x)
		return x
	def forward(self,inputs):
		x = self.enc_stack(inputs)
		x = x.flatten(start_dim=1)
		z = self.enc_logits(x)
		z = self.dec_stack(z) # to input shape
		z = z.view((-1,self.img_channels,self.img_height,self.img_width))
		# go through residual blocks
		z = self.resnet_pass(z)
		# post processing
		xre_logits = self.dec_post_proc(z)
		# output the logits, 
		return [xre_logits,inputs,z]
	def calc_loss(self,forward_out,**kwargs):
		use_sigmoid = kwargs.get("sigmoid",False)
		xre = forward_out[0]
		if use_sigmoid:
			xre = xre.sigmoid()
		xin = forward_out[1]
		output_dim = np.prod(xin.size()[1:])
		zin = forward_out[2]
		if self.use_mse:
			rec_loss = F.mse_loss(xre.view(-1,output_dim),xin.view(-1,output_dim),reduction="none")
			rec_loss = torch.mean(torch.sum(rec_loss,dim=1),dim=0)
		else:
			rec_loss = F.binary_cross_entropy_with_logits(xre_logits.view(-1,output_dim),xin.view(-1,output_dim),reduction='none')
			rec_loss = torch.mean(torch.sum(rec_loss,dim=1),dim=0)
		return {
			"loss":rec_loss,
			"reconstruction":rec_loss,
			}
		return {}
	def decode(self,zin):
		z = self.dec_stack(zin)
		z = z.view((self.img_channels,self.img_height,self.img_width))
		z = self.resnet_pass(z)
		xre = self.dec_post_proc(z)
		if not self.use_mse:
			xre = xre.sigmoid()
		return [xre.detach(),zin.detach()]