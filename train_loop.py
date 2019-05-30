import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

import os

from utils import compute_eer

class TrainLoop(object):

	def __init__(self, model_s, model_l, optimizer_s, optimizer_l, train_loader, valid_loader, patience, verbose=-1, cp_name=None, save_cp=False, checkpoint_path=None, checkpoint_epoch=None, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, cp_name) if cp_name else os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.model_s = model_s
		self.model_l = model_l
		self.optimizer_s = optimizer_s
		self.optimizer_l = optimizer_l
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.total_iters = 0
		self.cur_epoch = 0
		self.verbose = verbose
		self.save_cp = save_cp
		self.history = {'ce_asv': [], 'ce_asv_batch': [], 'ce_lid': [], 'ce_lid_batch': [], 'valid_loss': []}

		self.scheduler_s = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_s, factor=0.5, patience=patience, verbose=True if self.verbose>0 else False, threshold=1e-4, min_lr=1e-7)
		self.scheduler_l = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_l, factor=0.5, patience=patience, verbose=True if self.verbose>0 else False, threshold=1e-4, min_lr=1e-7)

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1, save_every=1):

		while (self.cur_epoch < n_epochs):

			np.random.seed()

			if self.verbose>0:
				print(' ')
				print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
				train_iter = tqdm(enumerate(self.train_loader))
			else:
				train_iter = enumerate(self.train_loader)

			ce_asv_epoch=0.0
			ce_lid_epoch=0.0
			for t, batch in train_iter:
				ce_asv, ce_lid = self.train_step(batch)
				self.history['ce_asv_batch'].append(ce_asv)
				self.history['ce_lid_batch'].append(ce_lid)
				ce_asv_epoch+=ce_asv
				ce_lid_epoch+=ce_lid
				self.total_iters += 1

			self.history['ce_asv'].append(ce_asv_epoch/(t+1))
			self.history['ce_lid'].append(ce_lid_epoch/(t+1))

			if self.verbose>0:
				print('Total train loss, ASV CE, and LID CE: {:0.4f}, {:0.4f}, {:0.4f}'.format(self.history['ce_asv'][-1]+self.history['ce_lid'][-1], self.history['ce_asv'][-1], self.history['ce_lid'][-1]))



			scores, labels = None, None

			for t, batch in enumerate(self.valid_loader):
				scores_batch, labels_batch = self.valid(batch)

				try:
					scores = np.concatenate([scores, scores_batch], 0)
					labels = np.concatenate([labels, labels_batch], 0)
				except:
					scores, labels = scores_batch, labels_batch

			self.history['valid_loss'].append(compute_eer(labels, scores))
			if self.verbose>0:
				print('Current validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_loss'][-1], np.min(self.history['valid_loss']), 1+np.argmin(self.history['valid_loss'])))

			self.scheduler_s.step(self.history['valid_loss'][-1])
			self.scheduler_l.step(self.history['valid_loss'][-1])

			if self.verbose>0:
				print('Current LR: {}'.format(self.optimizer.param_groups[0]['lr']))

			self.cur_epoch += 1

			if self.save_cp and (self.cur_epoch % save_every == 0 or self.history['valid_loss'][-1] < np.min([np.inf]+self.history['valid_loss'][:-1])):
					self.checkpointing()

		if self.verbose>0:
			print('Training done!')

		if self.verbose>0:
			print('Best validation loss and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['valid_loss']), 1+np.argmin(self.history['valid_loss'])))

		return np.min(self.history['valid_loss'])


	def train_step(self, batch):

		self.model_s.train()
		self.model_l.train()
		self.optimizer_s.zero_grad()
		self.optimizer_l.zero_grad()


		utterances, y_s, y_l = batch
		utterances.resize_(utterances.size(0)*utterances.size(1), utterances.size(2), utterances.size(3), utterances.size(4))
		y_s.resize_(y_s.numel())
		y_l.resize_(y_l.numel())

		ridx = np.random.randint(utterances.size(3)//4, utterances.size(3))
		utterances = utterances[:,:,:,:ridx]

		if self.cuda_mode:
			utterances = utterances.cuda(self.model_s.device)
			y_s = y_s.cuda(self.model_s.device).squeeze()
			y_l = y_l.cuda(self.model_s.device).squeeze()

		embeddings_l, h, c = self.model_l.forward(utterances)
		embeddings_s = self.model_s.forward(utterances, h, c)

		embeddings_norm_l = F.normalize(embeddings_l, p=2, dim=1)
		embeddings_norm_s = F.normalize(embeddings_s, p=2, dim=1)

		loss_asv = F.cross_entropy(self.model_s.out_proj(embeddings_norm_s, y_s), y_s)
		loss_lid = F.cross_entropy(self.model_l.out_proj(embeddings_norm_l, y_l), y_l)

		(loss_asv+loss_lid).backward()
		self.optimizer_s.step()
		self.optimizer_l.step()

		return loss_asv.item(), loss_lid.item()

	def valid(self, batch):

		self.model_l.eval()
		self.model_s.eval()

		with torch.no_grad():

			xa, xp, xn = batch

			ridx = np.random.randint(xa.size(3)//2, xa.size(3))

			xa, xp, xn = xa[:,:,:,:ridx], xp[:,:,:,:ridx], xn[:,:,:,:ridx]

			if self.cuda_mode:
				xa = xa.contiguous().cuda(self.model_s.device)
				xp = xp.contiguous().cuda(self.model_s.device)
				xn = xn.contiguous().cuda(self.model_s.device)

			emb_a_l, h, c = self.model_l.forward(xa)
			emb_a = self.model_s.forward(xa, h, c)
			emb_p_l, h, c = self.model_l.forward(xp)
			emb_p = self.model_s.forward(xp, h, c)
			emb_n_l, h, c = self.model_l.forward(xn)
			emb_n = self.model_s.forward(xn, h, c)

			scores_p = torch.nn.functional.cosine_similarity(emb_a, emb_p)
			scores_n = torch.nn.functional.cosine_similarity(emb_a, emb_n)

		return np.concatenate([scores_p.detach().cpu().numpy(), scores_n.detach().cpu().numpy()], 0), np.concatenate([np.ones(scores_p.size(0)), np.zeros(scores_n.size(0))], 0)

	def checkpointing(self):

		# Checkpointing
		if self.verbose>0:
			print('Checkpointing...')
		ckpt = {'model_s_state': self.model_s.state_dict(),
		'model_l_state': self.model_l.state_dict(),
		'optimizer_s_state': self.optimizer_s.state_dict(),
		'optimizer_l_state': self.optimizer_l.state_dict(),
		'scheduler_s_state': self.scheduler_s.state_dict(),
		'scheduler_l_state': self.scheduler_l.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch}
		try:
			torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))
		except:
			torch.save(ckpt, self.save_epoch_fmt)

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt, map_location = lambda storage, loc: storage)
			# Load model state
			self.model_s.load_state_dict(ckpt['model_s_state'])
			self.model_l.load_state_dict(ckpt['model_l_state'])
			# Load optimizer state
			self.optimizer_s.load_state_dict(ckpt['optimizer_s_state'])
			self.optimizer_l.load_state_dict(ckpt['optimizer_l_state'])
			# Load scheduler state
			self.scheduler_s.load_state_dict(ckpt['scheduler_s_state'])
			self.scheduler_l.load_state_dict(ckpt['scheduler_l_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			if self.cuda_mode:
				self.model = self.model.cuda(self.device)

		else:
			print('No checkpoint found at: {}'.format(ckpt))
