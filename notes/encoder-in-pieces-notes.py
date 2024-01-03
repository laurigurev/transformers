import torch
from torch import nn
from positionalencoder import PositionalEncoder
from multiheadattention import MultiHeadAttention

em = nn.Embedding(10, 2)
pos = PositionalEncoder(15, 2)
mha = MultiHeadAttention(2,2)
norm = nn.BatchNorm1d(2, eps=1e-5)
fc = nn.Linear(2,2)

input = torch.randint(0,9,(5,))
# print("input ", input.shape)
out = em(input)
# print("em    ", out.shape)
out = pos(out)
print('positional encoding: ', out.shape)
res = out
# print(res)
out = mha(out)
print('multihead attention: ', out.shape)
# print(out)
out += res
# print(out)
out = norm(out)
print('norm layer: ', out.shape)
# print(out)
out = fc(out)
print('linear shape: ', out.shape)