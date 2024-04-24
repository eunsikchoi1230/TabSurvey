import torch

if torch.cuda.is_available():
    print("CUDA is available on this system.")
else:
    print("CUDA is not available on this system.")

print(torch.cuda.device_count())
# import sklearn
# print(sklearn.__version__)