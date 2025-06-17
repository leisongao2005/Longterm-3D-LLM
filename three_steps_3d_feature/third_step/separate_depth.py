import os
import shutil
import json

SRC_PATH = "/local1/whu/data/hm3d/semantic_room_data/original_2d_gt_seg_multiple_x_0226_v6_0415_simple_180_scenes_18kx10_gemini20flash_v7"

# CURR_DATA_PATH = "/local1/leisongao/data/3dllm/rgb_features"

RBG_PATH = "/local1/leisongao/data/3dllm/rgb_features_eval"

DEPTH_PATH = "/local1/leisongao/data/3dllm/depth_features_eval"


def main():

    for task in os.listdir(RBG_PATH):
        # if not os.path.exists(os.path.join(DEPTH_PATH, task)):
        #     os.mkdir(os.path.join(DEPTH_PATH, task))
        
        for file in os.listdir(os.path.join(RBG_PATH, task)):
            if file[:5] == "depth" and file[-3:] == "png":
                os.remove(os.path.join(RBG_PATH, task, file))
                # shutil.move(os.path.join(RBG_PATH, task, file), os.path.join(DEPTH_PATH, task, file))
            
        os.remove(os.path.join(RBG_PATH, task, "points.npy"))

    # for scene in unprocessed:
    #     shutil.copytree(src=os.path.join(SRC_PATH, scene),
    #                 dst=os.path.join(NEW_DATA_PATH, scene))

if __name__=="__main__":
    main()

