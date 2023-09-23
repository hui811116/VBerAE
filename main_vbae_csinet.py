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
parser.add_argument("method",choices=["csinet",'vbae'],help="select the method for CsiNet dataset")
parser.add_argument("--epochs",type=int,help="epochs per run",default=10)
parser.add_argument("--batch_size",type=int,help="batch size for the simulation",default=64)
parser.add_argument("--learning_rate",type=float,help="learning rate for training",default=1e-3)
parser.add_argument("--latent_dim",type=int,help="number of clusters",default=128)
parser.add_argument("--save_dir",type=str,default="save_vbae_csinet",help="the path to save the models")
parser.add_argument("--data_path",type=str,default="../data/cost2100",help="default dataset path for download")
parser.add_argument("--seed",type=int,default=0,help="seed number of reproduction")
parser.add_argument("--reproduce",action="store_true",default=False,help="enable seeding")
parser.add_argument("--cpu",action="store_true",default=False,help="force using CPU")
parser.add_argument("--debug",action="store_true",default=False,help="debugging mode, load 10 data samples only")

args = parser.parse_args()
argsdict = vars(args)
print(argsdict)

if args.reproduce:
	uts.setup_seed(args.seed)
device = uts.getDevice(args.cpu)

train_dataloader, val_dataloader, test_dataloader = dts.getCsiNetDataLoaders(args.batch_size,True,device,args.data_path,env="indoor",debug=args.debug)

for batch, X in enumerate(train_dataloader):
	break

d_input_shape = X.size()[1:]

dataset_specific_dict = {"use_mse":True}
if args.method == "vbae":
	model = mds.VBerCsiNet(latent_dim=args.latent_dim,input_dim=d_input_shape,device=device,**dataset_specific_dict).to(device)
elif args.method == "csinet":
	model = mds.CsiNet(latent_dim=args.latent_dim,input_dim=d_input_shape,device=device,**dataset_specific_dict).to(device)

uts.print_network(model)

def train_loop(dataloader, model,optimizer):
	size = len(dataloader.dataset)
	for batch, X in enumerate(dataloader):
		X = X.to(device)
		pred = model(X)
		loss_dict = model.calc_loss(pred,sigmoid=True)
		loss = loss_dict['loss']
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 1000 == 0:
			#loss, current = loss.item(), (batch+1) * len(X)
			status_tex = ",".join(["{:}:{:>10.6f}".format(k,v) for k,v in loss_dict.items()])
			print(status_tex)

def test_loop(dataloader,model):
	size = len(dataloader.dataset)
	num_batchs = len(dataloader)
	#test_loss, test_mse, test_kl= 0, 0, 0
	test_dict = {}
	with torch.no_grad():
		for X in dataloader:
			X = X.to(device)
			pred = model(X)
			loss_dict = model.calc_loss(pred,sigmoid=True)
			if len(test_dict) != len(loss_dict):
				for k,v in loss_dict.items():
					test_dict[k]= 0
			for k,v in loss_dict.items():
				test_dict[k] += v
	status_tex = ",".join(["{:}:{:>10.6f}".format(k,v/num_batchs) for k,v in test_dict.items()])
	print(status_tex)


optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)

for t in range(args.epochs):
	print(f"Epoch {t+1}\n")
	train_loop(train_dataloader,model,optimizer)
	if (t+1)%5==0:
		test_loop(test_dataloader,model)

# 
os.makedirs(args.save_dir,exist_ok=True)
base_fname = "{:}_bs{:}_ep{:}_ld{:}_lr{:.4e}".format(args.method,args.batch_size,args.epochs,args.latent_dim,args.learning_rate)
safe_fname = uts.getSafeSaveName(args.save_dir,base_fname,".pth")

torch.save(model.state_dict(),os.path.join(args.save_dir,safe_fname+".pth"))
result_pkl = {"config":argsdict} # FIXME: add more info if needed
with open(os.path.join(args.save_dir,safe_fname+'.pkl'),'wb') as fid:
	pickle.dump(result_pkl,fid)

print("save the model weights to {:}".format(os.path.join(args.save_dir,safe_fname+".pth")))
