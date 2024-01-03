import torch
from torch import nn
import math

class PositionalEncoder(nn.Module):
	def __init__(self, max_seq_len, d):
		super(PositionalEncoder, self).__init__()
		print('positional encoder init')

		self.pos_enc = torch.zeros(max_seq_len, d, requires_grad=False)

		for pos in range(max_seq_len):
			for i in range(0,d,2):
				self.pos_enc[pos, i] = math.sin(pos/(10000**((2*i)/d)))
				self.pos_enc[pos, i+1] = math.cos(pos/(10000**((2*i)/d)))

	def forward(self, input):
		#input format seq*vectorsize
		out = input * self.pos_enc[:input.shape[0],:]
		return out