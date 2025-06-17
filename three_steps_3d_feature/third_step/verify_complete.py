import os


TARGET_DIR="/local1/leisongao/data/3dllm/rgb_features"
def main():
    all_scenes = os.listdir(TARGET_DIR)

    bad_scenes = []
    for scene in all_scenes:
        if not (os.path.exists(os.path.join(TARGET_DIR, scene, "pcd_pos_39.pt"))):
            bad_scenes.append(scene)
    
    print("checked all in dir, bad scenes: ")
    print(bad_scenes)
    print(f"num bad scenes: {len(bad_scenes)}")

if __name__=="__main__":
    main()