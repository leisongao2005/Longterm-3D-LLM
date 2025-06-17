import os
import sys
import shutil
import torch
from tqdm import tqdm
import json

TARGET_DIR="/local1/leisongao/data/3dllm/rgb_features"
NUM_SAMPLES=40

def main():
    no_blip_list = []
    for scene in tqdm(sorted(os.listdir(TARGET_DIR)), desc="total scenes", position=0):
        print(f"inspecting scene {scene}")
        if not (os.path.exists(os.path.join(TARGET_DIR, scene, "pcd_feat.pt"))):
            no_blip_list.append(scene)
            print(f"no total pcd_feat for {scene}, adding to blip processing list")
            continue


        #### point cloud feature from the pcd_pos.pt file
        pc_feat = torch.tensor(
            torch.load(os.path.join(TARGET_DIR, scene, "pcd_feat.pt"), weights_only=False, map_location="cpu")
        )
        #### point cloud itself from the pcd_pos.pt file
        pc = torch.tensor(
            torch.load(os.path.join(TARGET_DIR, scene, "pcd_pos.pt"), weights_only=False, map_location="cpu")
        )

        for i in range(NUM_SAMPLES):
            idxes = torch.sort(torch.randperm(pc_feat.shape[0])[:5000])[1]
            pc_feat_sample = pc_feat[idxes]
            pc_sample = pc[idxes]

            torch.save(pc_feat_sample, os.path.join(TARGET_DIR, scene, f"pcd_feat_{i}.pt"))
            torch.save(pc_sample, os.path.join(TARGET_DIR, scene, f"pcd_pos_{i}.pt"))


    with open("no_blip_list.json", 'w') as f:
        json.dump(no_blip_list, f)
    

if __name__=="__main__":
    main()