import torch

if torch.cuda.is_available():
    print("GPU 可用！")
    print("设备名称:", torch.cuda.get_device_name(0))
    print("CUDA 版本:", torch.version.cuda)
else:
    print("GPU 不可用。")