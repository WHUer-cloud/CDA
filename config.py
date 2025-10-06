import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_id", type=int, required=True)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="./datasets/mvtec/")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/")
    parser.add_argument("--test_interval", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=256)
    return parser.parse_args()

OBJ_NAMES = [
    "bottle", "cable", 
    "capsule",
     "carpet", 
    "grid", "hazelnut", "leather",
    "metal_nut", 
    "pill", "screw", 
    "tile", "toothbrush", "transistor", 
    "wood", 
    "zipper"
]

# OBJ_NAMES = [
    # '01'
    # '02'
    # '03'
# ]

# OBJ_NAMES = ['candle', 
# 'capsules', 
# 'cashew', 'chewinggum', 'fryum', 'macaroni1',
# 'macaroni2'
# 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
# ]