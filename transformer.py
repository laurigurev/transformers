import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
	def __init__(self, vocab_size, vector_size=512, sequence_length=50):
		super(Transformer, self).__init__()

		self.encoder = Encoder(vocab_size, vector_size, sequence_length)
		self.decoder = Decoder(vocab_size, vector_size, sequence_length)

	def forward(self, input):
		hidden = self.encoder(input)
		out = self.decoder(input, hidden)
		return out