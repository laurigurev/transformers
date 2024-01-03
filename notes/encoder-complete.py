from encoder import Encoder
import torch

input = torch.randint(0,9,(5,))
model = Encoder(10, 6)
out = model(input)
print(out)