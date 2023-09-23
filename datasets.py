import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision import transforms
import os
import sys
import numpy as np
from torchnet.dataset import TensorDataset, ResampleDataset
import pickle
import scipy.io as sio

def getMnistDataLoaders(batch_size, shuffle=True, device="cuda",data_path="./data"):
	kwargs = {"num_workers":1, "pin_memory": True} if device == "cuda" else {}
	tx = transforms.ToTensor()
	train = DataLoader(datasets.MNIST(data_path, train=True, download=True, transform=tx),
						batch_size=batch_size, shuffle=shuffle, **kwargs)
	test = DataLoader(datasets.MNIST(data_path, train=False, download=True, transform=tx),
						batch_size=batch_size, shuffle=shuffle, **kwargs)
	return train, test
def getSvhnDataLoaders(batch_size, shuffle=True, device="cuda",data_path="./data"):
	kwargs = {"num_workers":1, "pin_memory": True} if device == "cuda" else {}
	tx = transforms.ToTensor()
	train = DataLoader(datasets.SVHN(data_path, split='train', download=True, transform=tx),
						batch_size=batch_size, shuffle=shuffle, **kwargs)
	test = DataLoader(datasets.SVHN(data_path, split="test", download=True, transform=tx),
						batch_size=batch_size, shuffle=shuffle, **kwargs)
	return train, test


class BinaryMnistDataset(Dataset):
	def __init__(self,train=False,transform=None,target_transform=None,root_dir='./data'):
		self.transform = transform
		self.target_transform = target_transform
		self.train = train
		tx = transforms.ToTensor()
		if train:
			self.orig_dataset = datasets.MNIST(root_dir, train=True, download=True, transform=tx)
		else:
			self.orig_dataset = datasets.MNIST(root_dir, train=False, download=True, transform=tx)
	def __len__(self):
		return len(self.orig_dataset)
	def __getitem__(self,idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		x,y = self.orig_dataset[idx]
		if self.transform:
			x = self.transform(x)
		# make the output binary
		x = x > 0.5 # mnist is normalized
		x = x.float()

		if self.target_transform:
			y = self.target_transform(y)
		return x,y

def getBinaryMnistDataLoaders(batch_size,shuffle=True,device="cuda",data_path="./data"):
	kwargs = {"num_workers":1, "pin_memory": True} if device == "cuda" else {}
	train = DataLoader(BinaryMnistDataset(train=True),
						batch_size=batch_size, shuffle=shuffle, **kwargs)
	test = DataLoader(BinaryMnistDataset(train=False),
						batch_size=batch_size, shuffle=shuffle, **kwargs)
	return train, test


def loading_audio_write_npy(npy_path,train=True,binary_mnist=False):
	# according to the document, the shapes are:
	# written digit: 28 * 28
	# MFCC: 39 * 13
	if train:
		with open(npy_path+"/data_wr_train.npy",'rb') as fid:
			x_mnist_train = np.load(fid)
			if binary_mnist:
				x_mnist_train = (x_mnist_train>0.5).astype('float32')
			x_mnist_train= np.reshape(x_mnist_train,(-1,1,28,28)).astype("float32")
		with open(npy_path+"/labels_train.npy",'rb') as fid:
			y_train = np.load(fid)
			y_train = y_train.astype("int32")
		with open(npy_path+"/data_sp_train.npy",'rb') as fid:
			x_audio_train = np.load(fid)
			x_audio_train = np.reshape(x_audio_train,(-1,1,39,13)).astype("float32")
		return {"view1":x_mnist_train, "view2":x_audio_train, "ylabel":y_train}
	else:
		with open(npy_path+"/data_wr_test.npy",'rb') as fid:
			x_mnist_test = np.load(fid)
			if binary_mnist:
				x_mnist_test = (x_mnist_test>0.5).astype('float32')
			x_mnist_test = np.reshape(x_mnist_test,(-1,1,28,28)).astype("float32")
		with open(npy_path+"/labels_test.npy",'rb') as fid:
			y_test = np.load(fid)
			y_test = y_test.astype("int32")
		with open(npy_path+"/data_sp_test.npy",'rb') as fid:
			x_audio_test = np.load(fid)
			x_audio_test = np.reshape(x_audio_test,(-1,1,39,13)).astype('float32')
		return {"view1":x_mnist_test, "view2":x_audio_test, "ylabel":y_test}


def fmnist1V(transform=None):
	train_data = datasets.FashionMNIST(
			root="data",
			train=True,
			download=True,
			transform=transform,
		)
	test_data = datasets.FashionMNIST(
			root="data",
			train=False,
			download=False,
			transform=transform,
		)
	return train_data, test_data


def loadCsiNetDataset(data_path,env,category,debug):
	if env=="indoor":
		if category == "train":
			mat = sio.loadmat(os.path.join(data_path,"DATA_Htrainin.mat"))
		elif category == "val":
			mat = sio.loadmat(os.path.join(data_path,"DATA_Hvalin.mat"))
		elif category == "test":
			mat = sio.loadmat(os.path.join(data_path,"DATA_Htestin.mat"))
		else:
			sys.exit("undefined dataset category {:}".format(category))
	elif env == "outdoor":
		if category == "train":
			mat = sio.loadmat(os.path.join(data_path,"DATA_Htrainout.mat"))
		elif category == "val":
			mat = sio.loadmat(os.path.join(data_path,"DATA_Hvalout.mat"))
		elif category == "test":
			mat = sio.loadmat(os.path.join(data_path,"DATA_Htestout.mat"))
		else:
			sys.exit("undefined dataset category {:}".format(category))
	else:
		sys.exit("undefined environment, please choose from [indoor/outdoor]")
	x_data = mat['HT']
	
	img_height = 32
	img_width = 32
	img_channels =2
	img_total = img_height * img_width * img_channels
	x_data = x_data.astype("float32")

	#channel first

	x_data = np.reshape(x_data, (len(x_data),img_channels,img_height,img_width))

	# debugging, pick only 10 samples
	if debug:
		debug_num = 10 
		x_data = x_data[:debug_num]
	#print(x_data.shape)
	#sys.exit()
	return {
		"x_data":x_data,
		"height":img_height,
		"width":img_width,
		"channels":img_channels,
		"format":"channels_first",
	}

class CsiNetDataset(Dataset):
	def __init__(self,category=False,transform=None,target_transform=None,root_dir='./data',env="indoor",debug=False):
		self.transform = transform
		self.target_transform = target_transform
		self.category = category
		tx = transforms.ToTensor()
		self.orig_dataset = torch.from_numpy(loadCsiNetDataset(root_dir,env,category,debug)['x_data'])
	def __len__(self):
		return len(self.orig_dataset)
	def __getitem__(self,idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		x = self.orig_dataset[idx]
		if self.transform:
			x = self.transform(x)
		return x

def getCsiNetDataLoaders(batch_size, shuffle=True, device="cuda",data_path="./data",env="indoor",debug=False):
	kwargs = {"num_workers":1, "pin_memory": True} if device == "cuda" else {}
	#tx = transforms.ToTensor()
	tx = None
	train = DataLoader(CsiNetDataset(category="train",transform=tx,root_dir=data_path,env=env,debug=debug),
						batch_size=batch_size, shuffle=shuffle, **kwargs)
	valid = DataLoader(CsiNetDataset(category="val",transform=tx,root_dir=data_path,env=env,debug=debug),
						batch_size=batch_size, shuffle=shuffle, **kwargs)
	test = DataLoader(CsiNetDataset(category="test",transform=tx,root_dir=data_path,env=env,debug=debug),
						batch_size=batch_size, shuffle=False, **kwargs)
	return train, valid, test