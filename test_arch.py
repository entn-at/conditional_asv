from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import model as model_

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
args = parser.parse_args()


batch = torch.rand(3, 1, args.ncoef, 200)
model_s = model_.ResNet_lstm(n_z=args.latent_size, ncoef=args.ncoef, proj_size=10)
model_l = model_.ResNet_mfcc(n_z=args.latent_size, ncoef=args.ncoef, proj_size=20)
mu_l, h, c = model_l(batch)
print(mu_l.size(), h.size(), c.size())
mu = model_s.forward(batch, h, c)
print(mu.size())
