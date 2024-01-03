from encoder import Encoder
from decoder import Decoder
from transformer import Transformer
import torch

input = torch.randint(0,9,(5,))
enc = Encoder(10, 6)
out = enc(input)
print("hidden shape: ", out.shape)

dec = Decoder(vocab_size=10, vector_size=6, sentence_length=5)
out = dec(input, out)
print(out)

model = Transformer(vocab_size=10, vector_size=6, sequence_length=5)
out = model(input)
print(out)