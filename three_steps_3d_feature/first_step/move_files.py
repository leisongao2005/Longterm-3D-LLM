import os
import sys
import shutil
from tqdm import tqdm
import json

WENBO_DIR="/local1/whu/data/hm3d/semantic_room_data/original_2d_gt_seg_multiple_x_0226_v6_0415_simple_180_scenes_18kx10_gemini20flash_v7"

TARGET_DIR="/local1/leisongao/data/3dllm/"

def main():
    with open("/home/leisongao/Longterm-3D-LLM/eval_unprocessed_rooms_new.json", "r") as f:
        scenes = json.load(f)

    print(scenes)
    for scene in tqdm(scenes):
        if not os.path.isdir(os.path.join(WENBO_DIR, scene)):
            print("not in here!")
            continue
        # os.makedirs(os.path.join(TARGET_DIR, scene), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, "rgb_features_eval", scene), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, "depth_features_eval", scene), exist_ok=True)

        for f in os.listdir(os.path.join(WENBO_DIR, scene)):
            if f[-4:] == ".jpg" or f[-5:] == ".json":
                shutil.copy(os.path.join(WENBO_DIR, scene, f), os.path.join(TARGET_DIR, "rgb_features_eval", scene))
            if f[-4:] == ".npy" and f != "points.npy":
                shutil.copy(os.path.join(WENBO_DIR, scene, f), os.path.join(TARGET_DIR, "depth_features_eval", scene))


if __name__=="__main__":
    main()