import os
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import copy
import random


def getDevice(force_cpu):
	try:
		if force_cpu:
			device= torch.device("cpu")
			print("force using CPU")
		elif torch.backends.mps.is_available():
			device = torch.device("mps")
			print("using Apple MX chipset")
		elif torch.cuda.is_available():
			device = torch.device("cuda")
			print("using Nvidia GPU")
		else:
			device = torch.device("cpu")
			print("using CPU")
		return device
	except:
		print("MPS is not supported for this version of PyTorch")
		if torch.cuda.is_available():
			device = torch.device("cuda")
			print("using Nvidia GPU")
		else:
			device = torch.device("cpu")
			print("using CPU")
		return device


def setup_seed(seed):
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True

def getSafeSaveName(savepath,basename,extension=".pkl"):
	repeat_cnt =0
	safename = copy.copy(basename)
	while os.path.isfile(os.path.join(savepath,safename+extension)):
		repeat_cnt += 1
		safename = "{:}_{:}".format(basename,repeat_cnt)
	# return without extension
	return safename
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)