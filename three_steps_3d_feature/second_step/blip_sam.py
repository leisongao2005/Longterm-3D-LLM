import torch
# import torchvision
import cv2
import numpy as np
from tqdm import tqdm
import os
from torch import nn
from lavis.models.eva_vit import create_eva_vit_g
import argparse
import time
import json

import torch.multiprocessing as mp

LOAD_IMG_HEIGHT = 512
LOAD_IMG_WIDTH = 512


def get_bbox_around_mask(mask):
    # mask: (img_height, img_width)
    # compute bbox around mask
    bbox = None
    nonzero_inds = torch.nonzero(mask)  # (num_nonzero, 2)
    if nonzero_inds.numel() == 0:
        topleft = [0, 0]
        botright = [mask.shape[0], mask.shape[1]]
        bbox = (topleft[0], topleft[1], botright[0], botright[1])  # (x0, y0, x1, y1)
    else:
        topleft = nonzero_inds.min(0)[0]  # (2,)
        botright = nonzero_inds.max(0)[0]  # (2,)
        bbox = (topleft[0].item(), topleft[1].item(), botright[0].item(), botright[1].item())  # (x0, y0, x1, y1)
    # x0, y0, x1, y1
    return bbox, nonzero_inds
        
def worker(gpu_id, worker_id, scenes_for_this_worker, args):
    import torchvision

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    # device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # torch.autograd.set_grad_enabled(False)
    # print(f"Worker using GPU {gpu_id}")
    # print(f"GPU {gpu_id} memory before model: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")
    

    visual_encoder = create_eva_vit_g(512).to(device)

    # print(f"GPU {gpu_id} memory after model: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")
    # print(f"Model parameters: {sum(p.numel() for p in visual_encoder.parameters())}")
    
    # print(f"Worker {gpu_id} initialized with device: {device}")


    for scene in tqdm(scenes_for_this_worker, position=(2*gpu_id + worker_id-1), desc=f"Worker {worker_id} on gpu {gpu_id+1}", dynamic_ncols=True):
        try:
            os.makedirs(os.path.join(args.save_dir_path, scene), exist_ok=True)

            # print(f"inspecing scene {scene}")

            for file in os.listdir(os.path.join(args.mask_dir_path, scene)):
                INPUT_IMAGE_PATH = os.path.join(args.scene_dir_path, scene, file.replace(".pt", ".jpg"))
                SEMIGLOBAL_FEAT_SAVE_FILE = os.path.join(args.save_dir_path, scene, file)

                if os.path.isfile(SEMIGLOBAL_FEAT_SAVE_FILE):
                    continue

                raw_image = cv2.imread(INPUT_IMAGE_PATH)
                raw_image = cv2.resize(raw_image, (512, 512))
                image = torch.tensor(raw_image[:512, :512]).permute(2, 0, 1).unsqueeze(0).to(device)

                
                with torch.amp.autocast("cuda"):
                    output = visual_encoder(image)

                # global_feat = torch.tensor(output)
                global_feat = output.detach().clone()
                global_feat = global_feat.half().cuda()
                global_feat = global_feat.mean(1)
                # global_feat = global_feat[:, :-1, :].resize(1, 36, 36, 1408).permute((0, 3, 1, 2))
                # m = nn.AdaptiveAvgPool2d((1, 1))
                # global_feat = m(global_feat)
                # global_feat = global_feat.squeeze(-1).squeeze(-1)

                global_feat = torch.nn.functional.normalize(global_feat, dim=-1)
                FEAT_DIM = global_feat.shape[-1]


                cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

                MASK_LOAD_FILE = os.path.join(args.mask_dir_path, scene, file)
                outfeat = torch.zeros(512, 512, FEAT_DIM, dtype=torch.half)

                # print(f"Loading instance masks {MASK_LOAD_FILE}...")
                mask = torch.load(MASK_LOAD_FILE).unsqueeze(0)  # 1, num_masks, H, W

                mask = mask[:, :, :512, :512]
                num_masks = mask.shape[-3]

                rois = []
                roi_similarities_with_global_vec = []
                roi_sim_per_unit_area = []
                feat_per_roi = []
                roi_nonzero_inds = []
                for _i in range(num_masks):
                    ts4 = time.time()
                    curmask = mask[0, _i].long()
                    bbox, nonzero_inds = get_bbox_around_mask(curmask)
                    x0, y0, x1, y1 = bbox

                    bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
                    img_area = LOAD_IMG_WIDTH * LOAD_IMG_HEIGHT
                    iou = bbox_area / img_area

                    if iou < 0.005:
                        continue
                    roi = torch.ones((512, 512, 3))
                    img_roi = torch.tensor(raw_image[:512, :512])[x0:x1, y0:y1]
                    roi[x0:x1, y0:y1] = img_roi
                    img_roi = roi.permute(2, 0, 1).unsqueeze(0).to(device)

                
                    with torch.amp.autocast("cuda"):
                        roifeat = visual_encoder(img_roi)
                        
                    # roifeat = torch.tensor(roifeat)
                    roifeat = roifeat.detach().clone()
                    roifeat = roifeat.half().cuda()
                    roifeat = roifeat.mean(1)
                    # roifeat = roifeat[:, :-1, :].resize(1, 36, 36, 1408).permute((0, 3, 1, 2))
                    # m = nn.AdaptiveAvgPool2d((1, 1))
                    # roifeat = m(roifeat)
                    # roifeat = roifeat.squeeze(-1).squeeze(-1)

                    roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
                    feat_per_roi.append(roifeat)
                    roi_nonzero_inds.append(nonzero_inds)

                    _sim = cosine_similarity(global_feat, roifeat)

                    rois.append(torch.tensor(list(bbox)))
                    roi_similarities_with_global_vec.append(_sim)
                    roi_sim_per_unit_area.append(_sim)

                rois = torch.stack(rois)
                scores = torch.cat(roi_sim_per_unit_area).to(rois.device)
                retained = torchvision.ops.nms(rois.float().cpu(), scores.float().cpu(), iou_threshold=1.0)
                feat_per_roi = torch.cat(feat_per_roi, dim=0)

                retained_rois = rois[retained]
                retained_scores = scores[retained]
                retained_feat = feat_per_roi[retained]
                retained_nonzero_inds = []
                for _roiidx in range(retained.shape[0]):
                    retained_nonzero_inds.append(roi_nonzero_inds[retained[_roiidx].item()])

                mask_sim_mat = torch.nn.functional.cosine_similarity(
                    retained_feat[:, :, None], retained_feat.t()[None, :, :]
                )
                mask_sim_mat.fill_diagonal_(0.0)
                mask_sim_mat = mask_sim_mat.mean(1)  # avg sim of each mask with each other mask
                softmax_scores = retained_scores.cuda() - mask_sim_mat
                softmax_scores = torch.nn.functional.softmax(softmax_scores, dim=0)
                for _roiidx in range(retained.shape[0]):
                    _weighted_feat = (
                        softmax_scores[_roiidx] * global_feat + (1 - softmax_scores[_roiidx]) * retained_feat[_roiidx]
                    )
                    _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
                    outfeat[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]] += (
                        _weighted_feat[0].detach().cpu().half()
                    )
                    outfeat[
                        retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]
                    ] = torch.nn.functional.normalize(
                        outfeat[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]].float(),
                        dim=-1,
                    ).half()

                outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
                outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
                outfeat = torch.nn.functional.interpolate(outfeat, [512, 512], mode="nearest")
                outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
                outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
                outfeat = outfeat[0].half()  # --> H, W, feat_dim

                torch.save(outfeat, SEMIGLOBAL_FEAT_SAVE_FILE)
        except Exception as e:
            print(f"Error processing {scene}: {e}")
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir_path", default="./masked_rdp_data/", type=str)
    parser.add_argument("--mask_dir_path", default="./sam_masks/", type=str)
    parser.add_argument("--save_dir_path", default="./nps_sam_blip/", type=str)
    args = parser.parse_args()
    
    NUM_GPUS = 4
    NUM_WORKERS = 2

    # with open("/home/leisongao/Longterm-3D-LLM/three_steps_3d_feature/third_step/no_blip_list.json", "r") as f:
    #     all_scenes = json.load(f)
    # all_scenes = os.listdir(args.mask_dir_path)

    # new_scenes = []

    # for scene in all_scenes:
    #     if os.path.exists(os.path.join(args.scene_dir_path, scene, "pcd_pos_0.pt")):
    #         continue
    #     if os.path.exists(os.path.join(args.save_dir_path, scene, "image_10.pt")):
    #         continue
    #     new_scenes.append(scene)


    # print(f"scenes to process: {len(new_scenes)}")
    # return

    all_scenes = sorted(os.listdir(args.scene_dir_path))
    scenes_per_gpu = len(all_scenes) // NUM_GPUS
    scenes_per_worker = scenes_per_gpu // NUM_WORKERS

    processes = []

    for gpu_id in range(NUM_GPUS):
        start_idx = gpu_id * scenes_per_gpu
        mid_idx = gpu_id * scenes_per_gpu + scenes_per_worker
        if gpu_id == NUM_GPUS - 1:
            end_idx = len(all_scenes)  # last GPU takes the remainder
        else:
            end_idx = (gpu_id + 1) * scenes_per_gpu
            
        scenes_for_worker1 = all_scenes[start_idx:mid_idx]
        scenes_for_worker2 = all_scenes[mid_idx:end_idx]

        p1 = mp.Process(target=worker, args=(gpu_id, 1, scenes_for_worker1, args))
        p1.start()
        processes.append(p1)
        
        p2 = mp.Process(target=worker, args=(gpu_id, 2, scenes_for_worker2, args))
        p2.start()
        processes.append(p2)

    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    torch.cuda.empty_cache()
    main()