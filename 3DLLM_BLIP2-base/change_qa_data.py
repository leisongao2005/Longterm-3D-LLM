import json
from tqdm import tqdm


def main():
    PATHS = [
        "/local1/whu/data/hm3d/train_data/llava-3d_data_Feb28_v6_medium_180_scenes_18x10k_gemini_annotation_TrainQA_longtermmem_34k_training_samples.json",
        "/local1/whu/data/hm3d/train_data/llava-3d_data_Feb28_v6_medium_180_scenes_18x10k_gemini_annotation_TrainQA_longtermmem_865_evaluation_samples.json"
    ]

    NEW_PATHS = [
        "llava-3d_data_Feb28_v6_medium_180_scenes_18x10k_gemini_annotation_TrainQA_longtermmem_34k_training_samples_fixed.json",
        "llava-3d_data_Feb28_v6_medium_180_scenes_18x10k_gemini_annotation_TrainQA_longtermmem_865_evaluation_samples_fixed.json"
    ]

    for path, new_path in zip(PATHS, NEW_PATHS):
        with open(path, "r") as f:
            data = json.load(f)

        for task in tqdm(data, desc=f"processing {new_path}"):
            prompt = task["conversations"][0]["value"]

            index = prompt.find("answer the following questions:")


            question = prompt[index + 32:]
            start = prompt[:537]

            # print(f"start: {start}")
            # print(f"question at idx {index}: {question}")

            new_prompt = start + " I have provided five room point clouds, each 32 tokens long, Please answer the following question based on the task, and trajectory we provide next: " + question + prompt[537: index + 32 - 63]
            # print(f"\nupdated prompt: {new_prompt}")

            task["conversations"][0]["value"] = new_prompt

        with open(new_path, "w") as f:
            json.dump(data, f, indent=4)

if __name__=="__main__":
    main()