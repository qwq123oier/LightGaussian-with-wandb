#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser

# 添加 wandb 导入
try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")

# 添加 wandb 参数
parser.add_argument("--use_wandb", action="store_true", default=True, help="启用wandb日志")
parser.add_argument("--wandb_project", type=str, default="LightGaussian-full", help="wandb项目名")
parser.add_argument("--wandb_name", type=str, default=None, help="wandb运行名（可选）")

# 添加修剪相关参数
parser.add_argument("--prune_iterations", nargs="+", type=int, default=[20_000], help="修剪操作的迭代次数")
parser.add_argument("--prune_percent", type=float, default=0.6, help="第一次修剪的百分比")
parser.add_argument("--prune_decay", type=float, default=0.6, help="后续修剪的衰减率")
parser.add_argument("--v_pow", type=float, default=0.1, help="体积重要性指数")

# 添加端口参数（可选）
parser.add_argument("--port", type=int, default=6009, help="网络GUI端口")

args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", default="/home/user/Desktop/xinzeli/dataset", type=str)
    parser.add_argument("--tanksandtemples", "-tat", default="/home/user/Desktop/xinzeli/dataset", type=str)
    parser.add_argument("--deepblending", "-db", default="/home/user/Desktop/xinzeli/dataset", type=str)
    args = parser.parse_args()

if not args.skip_training:
    common_args = " --quiet --eval --test_iterations -1 "
    
    # 添加修剪参数到训练命令
    prune_args = f" --prune_percent {args.prune_percent} --prune_decay {args.prune_decay} --v_pow {args.v_pow}"
    prune_iterations_str = " ".join(map(str, args.prune_iterations))
    prune_args += f" --prune_iterations {prune_iterations_str}"
    
    # 添加端口参数
    port_arg = f" --port {args.port}"
    
    # 添加 wandb 参数到训练命令
    if args.use_wandb:
        common_args += " --use_wandb --wandb_project " + args.wandb_project
    
    # 将修剪参数和端口参数添加到通用参数中
    common_args += prune_args
    


    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        scene_wandb_name = f"{scene}_outdoor_speedy" if args.use_wandb else ""
        wandb_arg = f" --wandb_name {scene_wandb_name}" if scene_wandb_name else ""
        print(f"训练场景: {scene} (户外)")
        print(f"命令: python train.py -s {source} -i images_4 -m {args.output_path}/{scene} {common_args} {wandb_arg}")
        os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + common_args + wandb_arg)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        scene_wandb_name = f"{scene}_indoor_speedy" if args.use_wandb else ""
        wandb_arg = f" --wandb_name {scene_wandb_name}" if scene_wandb_name else ""
        print(f"训练场景: {scene} (室内)")
        print(f"命令: python train.py -s {source} -i images_2 -m {args.output_path}/{scene} {common_args} {wandb_arg}")
        os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + common_args + wandb_arg)
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        scene_wandb_name = f"{scene}_tandt_speedy" if args.use_wandb else ""
        wandb_arg = f" --wandb_name {scene_wandb_name}" if scene_wandb_name else ""
        print(f"训练场景: {scene} (Tanks & Temples)")
        print(f"命令: python train.py -s {source} -m {args.output_path}/{scene} {common_args} {wandb_arg}")
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + wandb_arg)
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        scene_wandb_name = f"{scene}_deepblending_speedy" if args.use_wandb else ""
        wandb_arg = f" --wandb_name {scene_wandb_name}" if scene_wandb_name else ""
        print(f"训练场景: {scene} (Deep Blending)")
        print(f"命令: python train.py -s {source} -m {args.output_path}/{scene} {common_args} {wandb_arg}")
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + wandb_arg)

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)

    common_args = " --quiet --eval --skip_train"
    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)