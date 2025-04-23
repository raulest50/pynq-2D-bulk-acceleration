"""
To check if you have a GPU available, you can use the following code.
"""

import pyopencl as cl

gpu_devices = []
for plat in cl.get_platforms():
    for dev in plat.get_devices():
        if dev.get_info(cl.device_info.TYPE) == cl.device_type.GPU:
            gpu_devices.append(dev)
if not gpu_devices:
    print("No GPU found")
else:
    print(f"Found {len(gpu_devices)} GPU(s):")
    for gpu in gpu_devices:
        print(" ", gpu.name)



