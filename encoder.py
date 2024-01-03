import torch
from torch import nn
from positionalencoder import PositionalEncoder
from multiheadattention import MultiHeadAttention

class Encoder(nn.Module):
	def __init__(self, vocab_size, vector_size=512, seq_len=50):
		super(Encoder, self).__init__()
		print('encoder init')

		self.embed = nn.Embedding(vocab_size, vector_size)
		self.pos_enc = PositionalEncoder(seq_len, vector_size)
		self.mha = MultiHeadAttention(vector_size, 4)
		self.norm = nn.BatchNorm1d(vector_size)
		self.fc1 = nn.Linear(vector_size, vector_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(vector_size, vector_size)

	def forward(self, input):
		out = self.embed(input)
		out = self.pos_enc(out)

		residual = out

		out = self.mha(out, out)
		out += residual

		residual = out

		out = self.norm(out)

		out = self.fc1(out)
		out = self.relu(out)
		out = self.fc2(out)

		out += residual
		out = self.norm(out)

		return out