import os
import shutil
import json

SRC_PATH = "/local1/whu/data/hm3d/semantic_room_data/original_2d_gt_seg_multiple_x_0226_v6_0415_simple_180_scenes_18kx10_gemini20flash_v7"

CURR_DATA_PATH = "/local1/leisongao/data/3dllm/rgb_features"

NEW_DATA_PATH = "/local1/leisongao/data/3dllm/rgb_features_eval"


def main():
    with open("/home/leisongao/Longterm-3D-LLM/eval_unprocessed_rooms.json", "r") as f:
        unprocessed = json.load(f)
    
    print(len(unprocessed))

    bad_rooms = []
    for img_id in unprocessed:
        if not (os.path.exists(os.path.join(CURR_DATA_PATH, img_id, "pcd_pos_39.pt"))):
            bad_rooms.append(img_id)

    print(len(bad_rooms))
    with open("/home/leisongao/Longterm-3D-LLM/eval_unprocessed_rooms_new.json", "w") as f:
        json.dump(bad_rooms, f)

if __name__=="__main__":
    new_scenes = main()


