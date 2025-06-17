import json

# OUTPUT_DIR = "/home/leisongao/Longterm-3D-LLM/3DLLM_BLIP2-base/lavis/output/BLIP2/3DCaption/20250504211/result/val_16_vqa_result.json"
OUTPUT_DIR = "/home/leisongao/Longterm-3D-LLM/3DLLM_BLIP2-base/lavis/output/BLIP2/longtermTask_Eval/eval_in_wild/result/val_75_vqa_result.json"


# DATA_DIR = "/local1/whu/data/hm3d/train_data/Captions/llava-3d_data_Feb26_v6_medium_13K_caption_episodic_mem_v2_natural_cap_sum_diff_evaluation_samples.json"

DATA_DIR = "/local1/whu/data/hm3d/train_data/llava-3d_data_0415_simple_180_scenes_18kx10_gemini20flash_v7_memv4_cot_wentrooms_loose_15k_unseen_objects_samples.json"


def main():
    with open(OUTPUT_DIR, "r") as f:
        outputs = json.load(f)

    with open(DATA_DIR, "r") as f:
        data = json.load(f)
    

    inf_out = []
    for output in outputs:
        q_id = output["question_id"]
        model_ans = output["answer"]

        for full_question in data:
            question = full_question["training_data"]
            if question["id"] == q_id:
                entry = {
                    "id": q_id,
                    "conversations": question["conversations"] + [{
                        "from": "3d-llm",
                        "value": model_ans
                    }]
                }
                inf_out.append(entry)
                break
        
    # print(inf_out)

    # with open("3d-llm_in_domain-eval.json", "w") as f:
    #     json.dump(inf_out, f, indent=4)
    with open("/local1/leisongao/data/3dllm/3d-llm_in_wild-eval.json", "w") as f:
        json.dump(inf_out, f, indent=4)

if __name__=="__main__":
    main()