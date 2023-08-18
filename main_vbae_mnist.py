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

parser = argparse.ArgumentParser()
parser.add_argument("--epochs",type=int,help="epochs per run",default=10)
parser.add_argument("--batch_size",type=int,help="batch size for the simulation",default=64)
parser.add_argument("--learning_rate",type=float,help="learning rate for training",default=1e-3)
parser.add_argument("--latent_dim",type=int,help="number of clusters",default=64)
parser.add_argument("--save_dir",type=str,default="save_vbae_models",help="the path to save the models")
parser.add_argument("--data_path",type=str,default="./data",help="default dataset path for download")
parser.add_argument("--seed",type=int,default=0,help="seed number of reproduction")
parser.add_argument("--reproduce",action="store_true",default=False,help="enable seeding")
parser.add_argument("--cpu",action="store_true",default=False,help="force using CPU")
parser.add_argument("--cnn_hidden_channels",type=int,default=16,help="Number of channels for the VAE-CNN hidden layers")

args = parser.parse_args()
argsdict = vars(args)
print(argsdict)

if args.reproduce:
	uts.setup_seed(args.seed)


#train_dataloader, test_dataloader = dts.getMnistDataLoaders(args.batch_size,True,args.data_path)
train_dataloader, test_dataloader = dts.getBinaryMnistDataLoaders(args.batch_size,True,args.data_path)


device = uts.getDevice(args.cpu)

for batch, (X,Y) in enumerate(train_dataloader):
	break

d_input_shape = X.size()[1:]


dataset_specific_dict = {"hidden_layer_channels":args.cnn_hidden_channels}
model = mds.VBerAE(latent_dim=args.latent_dim,input_dim=d_input_shape,device=device,**dataset_specific_dict).to(device)

uts.print_network(model)


def train_loop(dataloader, model,optimizer):
	size = len(dataloader.dataset)
	for batch, (X,Y) in enumerate(dataloader):
		X = X.to(device)
		pred = model(X)
		loss_dict = model.calc_loss(pred)
		loss = loss_dict['loss']
		kl = loss_dict['kl_divergence']
		bce = loss_dict['reconstruction']
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:
			loss, current = loss.item(), (batch+1) * len(X)
			print(f"loss: {loss:>7f}, bce: {bce:>7f}, kl: {kl:>7f}  [{current:>5d}/{size:>5d}]")

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


optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)

for t in range(args.epochs):
	print(f"Epoch {t+1}\n")
	train_loop(train_dataloader,model,optimizer)
	if (t+1)%5==0:
		test_loop(test_dataloader,model)

# 
os.makedirs(args.save_dir,exist_ok=True)
base_fname = "vbae_bs{:}_ep{:}_ld{:}_lr{:.4e}".format(args.batch_size,args.epochs,args.latent_dim,args.learning_rate)
safe_fname = uts.getSafeSaveName(args.save_dir,base_fname,".pth")

torch.save(model.state_dict(),os.path.join(args.save_dir,safe_fname+".pth"))
result_pkl = {"config":argsdict} # FIXME: add more info if needed
with open(os.path.join(args.save_dir,safe_fname+'.pkl'),'wb') as fid:
	pickle.dump(result_pkl,fid)

print("save the model weights to {:}".format(os.path.join(args.save_dir,safe_fname+".pth")))
