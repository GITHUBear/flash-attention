import torch

base = 1000000.0
rotary_dim = 128
max_position_embeddings = 4096

inv_freq = 1.0 / (base**(torch.arange(
            0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
print(inv_freq.shape)
t = torch.arange(max_position_embeddings, dtype=torch.float)
freqs = torch.einsum("i,j -> ij", t, inv_freq)
cos = freqs.cos()
sin = freqs.sin()
cache = torch.cat((cos, sin), dim=-1)
print(cache.shape)