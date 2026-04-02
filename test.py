# import torch
# from mamba_ssm.modules.mamba2 import Mamba2

# device = "cuda"
# d_model = 512
# headdim = 64
# # Для MIMO (8 голов): 8 * 64 = 512
# d_ssm = 512 

# try:
#     model = Mamba2(
#         d_model=d_model,
#         d_ssm=d_ssm,    # Это заменяет nheads в вашей версии
#         headdim=headdim,
#         expand=2
#     ).to(device).to(torch.bfloat16)

#     x = torch.randn(2, 64, d_model, device=device, dtype=torch.bfloat16)
    
#     with torch.no_grad():
#         y = model(x)
    
#     # Проверка количества голов внутри модели
#     actual_heads = model.d_ssm // model.headdim
#     print(f"✅ Mamba-2/3 (MIMO) работает!")
#     print(f"Количество голов (MIMO): {actual_heads}")
#     print(f"Размерность головы: {model.headdim}")

# except Exception as e:
#     print(f"❌ Ошибка: {e}")


#####################################

# import torch
# from mamba_ssm import Mamba3

# print("torch:", torch.__version__)
# print("torch cuda:", torch.version.cuda)
# print("cuda available:", torch.cuda.is_available())

# batch, length, dim = 1, 256, 128
# x = torch.randn(batch, length, dim, device="cuda", dtype=torch.bfloat16)

# model = Mamba3(
#     d_model=dim,
#     d_state=64,
#     headdim=32,
#     is_mimo=False,
#     chunk_size=64,
#     is_outproj_norm=False,
#     dtype=torch.bfloat16,
# ).to("cuda")

# with torch.no_grad():
#     y = model(x)

# print("output shape:", y.shape)
# assert y.shape == x.shape
# print("Mamba3 SISO OK")