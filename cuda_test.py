"""
Developed By Fajarlabs

"""

import torch

print("CUDA available:", torch.cuda.is_available())  # Should print True
print("CUDA version:", torch.version.cuda)            # Should print 12.2
print("Number of GPUs:", torch.cuda.device_count())   # Number of available GPUs
print("GPU Name:", torch.cuda.get_device_name(0))     # Name of the first GPU