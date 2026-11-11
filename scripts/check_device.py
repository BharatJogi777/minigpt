import torch
print("PyTorch:",torch.__version__)
print("MPS built:",torch.backends.mps.is_built())
print("MPS available:",torch.backends.mps.is_available())
device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
x=torch.randn(256,256,device=device); y=torch.randn(256,256,device=device)
print("Selected device:",device); print("Matrix result:",(x@y).shape); print("Device smoke test passed")
