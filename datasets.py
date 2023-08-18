import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision import transforms
import os
import sys
import numpy as np
from torchnet.dataset import TensorDataset, ResampleDataset
import pickle

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

