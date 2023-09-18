CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg ./experiments/net_d_0_4000.json &
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg ./experiments/net_d_4_4000.json &
CUDA_VISIBLE_DEVICES=2 python tools/train.py --cfg ./experiments/net_d_9_4000.json &
CUDA_VISIBLE_DEVICES=3 python tools/train.py --cfg ./experiments/net_d_8_4000.json &
CUDA_VISIBLE_DEVICES=4 python tools/train.py --cfg ./experiments/net_d_5_4000.json &
CUDA_VISIBLE_DEVICES=5 python tools/train.py --cfg ./experiments/net_d_2_4000.json &

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg ./experiments/net_d_3_4000.json &
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg ./experiments/net_d_1_4000.json &
CUDA_VISIBLE_DEVICES=3 python tools/train.py --cfg ./experiments/net_d_7_4000.json &
CUDA_VISIBLE_DEVICES=2 python tools/train.py --cfg ./experiments/net_d_6_4000.json

CUDA_VISIBLE_DEVICES=1 python tools/test.py --cfg ./experiments/net_d_10_4000.json
python tools/compute_params.py --cfg ./experiments/net_d_10_4000.json
CUDA_VISIBLE_DEVICES=0 python tools/extract_weight.py --cfg ./experiments/net_d_10_4000.json
CUDA_VISIBLE_DEVICES=1 python tools/create_heatmap.py --cfg ./experiments/net_d_10_4000.json
