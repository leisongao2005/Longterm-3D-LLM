import os
import json


JSON_PATHS = [
    '/local1/whu/data/hm3d/train_data/llava-3d_data_0415_simple_180_scenes_18kx10_gemini20flash_v7_memv4_cot_wentrooms_loose_15k_evaluation_samples.json',
    '/local1/whu/data/hm3d/train_data/llava-3d_data_0415_simple_180_scenes_18kx10_gemini20flash_v7_memv4_cot_wentrooms_loose_15k_unseen_objects_samples.json'
]


def main():
    unprocessed_rooms = set()
    processed_rooms = set(os.listdir(
        "/local1/leisongao/data/3dllm/rgb_features"
    ))
    for path in JSON_PATHS:
        with open(path, "r") as f:
            data = json.load(f)

            for task in data:
                scene_rooms = task["training_data"]["video"]
                for room in scene_rooms:
                    room = room[5:]
                    if room in processed_rooms:
                        continue
                    else:
                        unprocessed_rooms.add(room)
        
    with open("/home/leisongao/Longterm-3D-LLM/eval_unprocessed_rooms.json", "w") as f:
        json.dump(list(unprocessed_rooms), f)

if __name__=="__main__":
    main()
        
