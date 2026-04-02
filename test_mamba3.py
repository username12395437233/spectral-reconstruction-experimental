import torch
from mamba_ssm import Mamba3

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

batch, length, dim = 1, 256, 128
x = torch.randn(batch, length, dim, device="cuda", dtype=torch.bfloat16)

model = Mamba3(
    d_model=dim,
    d_state=64,
    headdim=32,
    is_mimo=False,
    chunk_size=64,
    is_outproj_norm=False,
    dtype=torch.bfloat16,
).to("cuda")

with torch.no_grad():
    y = model(x)

print("output shape:", y.shape)
assert y.shape == x.shape
print("Mamba3 SISO OK")