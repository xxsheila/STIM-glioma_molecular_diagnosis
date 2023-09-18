import os
DEVICE_IP = 78  # 机器编号
# GPU = [4,5,6,7]  # 使用的gpu的序号
GPU = [0,1,2,3]  # 使用的gpu的序号
MODE = "valid"  # or "train" or "valid"

PROGRAM_PER_GPU = 3  # 每个gpu放多少个程序
LEN_GPU = len(GPU)
PROGRAMS_PER_GROUP = LEN_GPU * PROGRAM_PER_GPU  # 每组跑多少个程序
with open(f"tune_{MODE}_{DEVICE_IP}.sh", 'w') as f:
	# PATH = f'experiments/experiments_{DEVICE_IP}'
	PATH = f'experiments'  # 配制文件路径
	files_name = [os.path.basename(p.path) for p in os.scandir(PATH)]
	for i, file_name in enumerate(files_name):
		print(f'CUDA_VISIBLE_DEVICES={GPU[i % LEN_GPU]} python tools/{MODE}.py --cfg ./{PATH}/{file_name} &', file=f)
		if (i+1) % PROGRAMS_PER_GROUP == 0:
			print('wait', file=f)
