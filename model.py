import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from losses import AMSoftmax, Softmax

class SelfAttention(nn.Module):
	def __init__(self, hidden_size):
		super(SelfAttention, self).__init__()

		#self.output_size = output_size
		self.hidden_size = hidden_size
		self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

		init.kaiming_uniform_(self.att_weights)

	def forward(self, inputs):

		batch_size = inputs.size(0)
		weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

		if inputs.size(0)==1:
			attentions = F.softmax(torch.tanh(weights), dim=1)
			weighted = torch.mul(inputs, attentions.expand_as(inputs))
		else:
			attentions = F.softmax(torch.tanh(weights.squeeze()),dim=1)
			weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

		noise = 1e-5*torch.randn(weighted.size())

		if inputs.is_cuda:
			noise = noise.cuda(inputs.get_device())

		avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)

		representations = torch.cat((avg_repr,std_repr),1)

		return representations

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.activation = nn.ELU()
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.activation(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.activation(out)

		return out

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.activation = nn.ELU()
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.activation(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.activation(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.activation(out)

		return out

class ResNet_mfcc(nn.Module):
	def __init__(self, n_z=256, layers=[3,4,6,3], block=Bottleneck, proj_size=0, ncoef=23, sm_type='none'):
		self.inplanes = 32
		super(ResNet_mfcc, self).__init__()

		self.conv1 = nn.Conv2d(1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn1 = nn.BatchNorm2d(32)
		self.activation = nn.ELU()
		
		self.layer1 = self._make_layer(block, 16, layers[0],stride=1)
		self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
		self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

		self.fc = nn.Linear(2*512,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_mu = nn.Linear(512, n_z)

		self.cond_proj_h = nn.Linear(n_z, 2*2*256)
		self.cond_proj_c = nn.Linear(n_z, 2*2*256)

		self.initialize_params()

		self.attention = SelfAttention(512)

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion) )

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
	
		x = self.conv1(x)
		x = self.activation(self.bn1(x))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2)

		stats = self.attention(x.permute(0,2,1).contiguous())

		fc = F.elu(self.lbn(self.fc(stats)))

		mu = self.fc_mu(fc)
		h = self.cond_proj_h(mu).view(2*2,-1,256)
		c = self.cond_proj_c(mu).view(2*2,-1,256)

		return mu, h, c

class ResNet_lstm(nn.Module):
	def __init__(self, n_z=256, layers=[3,4,6,3], block=Bottleneck, proj_size=0, ncoef=23, sm_type='none'):
		self.inplanes = 32
		super(ResNet_lstm, self).__init__()
	
		self.conv1 = nn.Conv2d(1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn1 = nn.BatchNorm2d(32)
		self.activation = nn.ELU()
		
		self.layer1 = self._make_layer(block, 16, layers[0],stride=1)
		self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
		self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

		self.lstm = nn.LSTM(512, 256, 2, bidirectional=True, batch_first=False)

		self.fc = nn.Linear(2*512+256,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_mu = nn.Linear(512, n_z)

		self.initialize_params()

		self.attention = SelfAttention(512)

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError



	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion) )

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x, h0, c0):
		x = self.conv1(x)
		x = self.activation(self.bn1(x))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2).permute(2,0,1)

		out_seq, (h_, c_) = self.lstm(x, (h0, c0))

		stats = self.attention(out_seq.permute(1,0,2).contiguous())

		x = torch.cat([stats,h_.mean(0)],dim=1)

		fc = F.elu(self.lbn(self.fc(x)))
		mu = self.fc_mu(fc)
		return mu