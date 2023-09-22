import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import models as mds
import datasets as dts
import sys
import os
import argparse
import utils as uts
import pickle
import matplotlib.pyplot as plt
import numpy as np

# load a trained model

# sample the model from testing data

# ## compute the rate-distortion function?
# ## by varying the number of bits for the representations

# sample the model from random bitstream

parser = argparse.ArgumentParser()
parser.add_argument("weight",type=str,help=".pth weight file")
#parser.add_argument("--data_path",type=str,default="./data",help="default dataset path for download")
parser.add_argument("--seed",type=int,default=0,help="seed number of reproduction")
parser.add_argument("--reproduce",action="store_true",default=False,help="enable seeding")
parser.add_argument("--cpu",action="store_true",default=False,help="force using CPU")

# 
args = parser.parse_args()
argsdict = vars(args)
print(argsdict)

if args.reproduce:
	uts.setup_seed(args.seed)



config_path = ".".join((args.weight).split(".")[:-1])+".pkl"
with open(config_path,'rb') as fid:
	config_dict = pickle.load(fid)

print("read config")
for k,v in config_dict.items():
	print("{:}={:}".format(k,v))
cfg = config_dict['config']

device = uts.getDevice(args.cpu)
#train_dataloader, test_dataloader = dts.getMnistDataLoaders(args.batch_size,True,args.data_path)
train_dataloader, test_dataloader = dts.getBinaryMnistDataLoaders(cfg['batch_size'],True,cfg['data_path'])
for batch, (X,Y) in enumerate(train_dataloader):
	break
d_input_shape = X.size()[1:]
# load model
dataset_specific_dict = {"hidden_layer_channels":cfg['cnn_hidden_channels']}
model = mds.VBerAE(latent_dim=cfg['latent_dim'],input_dim=d_input_shape,device=device,**dataset_specific_dict).to(device)

model.load_state_dict(torch.load(args.weight,map_location=device))

uts.print_network(model)

def test_loop(dataloader,model):
	size = len(dataloader.dataset)
	num_batchs = len(dataloader)
	test_loss, test_bce, test_kl= 0, 0, 0
	with torch.no_grad():
		for X,Y in dataloader:
			X = X.to(device)
			pred = model(X)
			loss_dict = model.calc_loss(pred)
			test_loss += loss_dict['loss']
			test_bce += loss_dict['reconstruction']
			test_kl += loss_dict['kl_divergence']
	test_loss /= num_batchs
	test_bce /= num_batchs
	test_kl /= num_batchs
	print(f"Test loss: {test_loss:>7f}, bce: {test_bce:>7f}, kl: {test_kl:>7f}\n")

optimizer = torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])

test_loop(test_dataloader,model) # get a sense of the performance

# AutoEncoder Reconstruction
nsamp = 100
eval_dataloader = DataLoader(dataset=test_dataloader.dataset,batch_size=nsamp)
for xin,yin in eval_dataloader:
	break
#xin,yin = test_dataloader.dataset[torch.arange(nsamp)] # NOTE: always pick the first nsamp
with torch.no_grad():
	xin = xin.to(device)
	pred = model(xin)
	xre = pred[0].view(tuple([-1]+list(d_input_shape))).sigmoid().cpu().numpy() # to normalied scaling
	xre = np.transpose(xre,axes=(0,2,3,1))

def plot_grayscale(x_in,suptitle):
	plt.figure(figsize=(20,8))
	nf = 10
	for ii in range(nf):
		for jj in range(nf):
			idx = ii + jj * nf
			ax = plt.subplot(nf,nf,idx+1)
			# channel last 
			xget = x_in[idx]
			plt.imshow(np.squeeze(xget),cmap='gray')
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.suptitle(suptitle)
	plt.show()

plot_grayscale(xre,'reconstructed')
plot_grayscale(xin.cpu().numpy(),'original')

# sample the output logits 