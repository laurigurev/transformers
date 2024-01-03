import torch
from torch import nn
from positionalencoder import PositionalEncoder
from multiheadattention import MultiHeadAttention

class Decoder(nn.Module):
	def __init__(self, vocab_size, vector_size=512, sentence_length=50):
		super(Decoder, self).__init__()
		print('decoder init')

		self.embed = nn.Embedding(vocab_size, vector_size)
		self.pos_enc = PositionalEncoder(sentence_length, vector_size)

		self.mha = MultiHeadAttention(vector_size, 4)
		self.norm = nn.BatchNorm1d(vector_size)

		self.fc1 = nn.Linear(vector_size, vector_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(vector_size, vector_size)

		self.linear = nn.Linear(vector_size*sentence_length, vocab_size)

		self.softmax = nn.Softmax(dim=-1)

	def forward(self, input, hidden):
		out = self.embed(input)
		out = self.pos_enc(out)

		residual = out

		out = self.mha(out, out, mask=True)
		out += residual
		residual = out

		out = self.norm(out)
		out = self.mha(hidden, out)

		out += residual
		residual = out
		out = self.norm(out)

		#feed forward
		out = self.fc1(out)
		out = self.relu(out)
		out = self.fc2(out)

		out += residual
		out = self.norm(out)

		#linear

		out = out.view(1, out.shape[0] * out.shape[1])
		out = self.linear(out)

		#softmax
		out = self.softmax(out)

		return out