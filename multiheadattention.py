import torch
from torch import nn
import math

class MultiHeadAttention(nn.Module):
	def __init__(self, in_features, num_heads=1):
		super(MultiHeadAttention, self).__init__()
		print('multiheadattentionlayer init')

		self.num_heads = num_heads

		self.query = nn.Linear(in_features, in_features * num_heads)
		self.key = nn.Linear(in_features, in_features * num_heads)
		self.value = nn.Linear(in_features, in_features * num_heads)
		self.softmax = nn.Softmax(dim=-1)
		self.concat = nn.Linear(num_heads * in_features, in_features)

	def forward(self, query_key_input, value, mask=False):
		q = self.query(query_key_input.detach().clone())
		k = self.key(query_key_input.detach().clone())
		v = self.value(value.detach().clone())

		# reshape
		q = torch.reshape(q, (self.num_heads, q.shape[0], int(q.shape[1] / self.num_heads)))
		k = torch.reshape(k, (self.num_heads, k.shape[0], int(k.shape[1] / self.num_heads)))
		v = torch.reshape(v, (self.num_heads, v.shape[0], int(v.shape[1] / self.num_heads)))

		# scaled dot product
		k = k.transpose(-1,-2)
		out = torch.bmm(q, k)
		out = out / math.sqrt(out.shape[1])
		# create a mask
		if mask == True:
			out = self.making_a_mask(out)
		# continuing sdp
		out = self.softmax(out)
		out = torch.mul(q, v)

		# concat
		out = torch.reshape(out, (out.shape[1], out.shape[2] * self.num_heads))
		out = self.concat(out)

		return out
	
	def making_a_mask(self, input):
		mask = torch.ones(input.shape[1], input.shape[1])
		for i in range(mask.shape[0]):
			for j in range(i):
				mask[j, i] = 0.
		out = input * mask
		return out