"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import torch
import numpy as np

from PIL import Image
from PIL import ImageFile
from collections import OrderedDict

ImageFile.LOAD_TRUNCATED_IMAGES = True

import time

# from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from lavis.datasets.datasets.base_dataset import BaseDataset


# assume this is never called
class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                # "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "pc_feat": sample["pc_feat"],
                "pc": sample["pc"],
                "answer": "; ".join(ann["answers"]),
            }
        )


class LongtermCaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.pc_feat_root = "/local1/leisongao/data/3dllm/rgb_features"  
        self.voxel_root = "/local1/leisongao/data/3dllm/rgb_features" # default point clouds

        self.scene_ids = {}
        n = 0
        new_annotation = []
        print(f"\nTrain Dataset there are {len(self.annotation)} possible examples")
        for ann in self.annotation:
            try:
                valid_ann = True
                for img_id in ann["video"]:
                    img_id = img_id[5:] # only get the last video
                    if img_id not in self.scene_ids.keys():
                        self.scene_ids[img_id] = n
                        n += 1
                    if not (os.path.exists(os.path.join(self.pc_feat_root, img_id, "pcd_pos_39.pt"))):
                        valid_ann = False
                if valid_ann:
                    new_annotation.append(ann)
            except:
                pass

        self.annotation = new_annotation

        print(f"\nTrain Dataset there are {len(self.annotation)} valid examples")

    def __getitem__(self, index):

        ann = self.annotation[index]

        captioning_prompt = self.text_processor(ann["conversations"][0]["value"])
        ##### ********* assumed that for captioning it is always only human --> gpt in pairs
        caption = ann["conversations"][1]["value"]

        pc_feat_list = []
        pc_list = []
        scene_id_list = []

        for scene in ann["video"]:
            scene_id = scene[5:]

            sample_id = torch.randint(40, ()).item()
            #### point cloud feature from the pcd_pos.pt file
            pc_feat = torch.tensor(
                torch.load(os.path.join(self.pc_feat_root, scene_id, f"pcd_feat_{sample_id}.pt"), weights_only=False, map_location="cpu")
            )
            #### point cloud itself from the pcd_pos.pt file
            pc = torch.tensor(
                torch.load(os.path.join(self.voxel_root, scene_id, f"pcd_pos_{sample_id}.pt"), weights_only=False, map_location="cpu")
            )

            pc_feat_list.append(pc_feat)
            pc_list.append(pc)
            scene_id_list.append(scene_id)


        answer_weight = { 
            caption: 1
        }

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        out = {
            "pc_feat": pc_feat_list,
            "pc": pc_list,
            "text_input": captioning_prompt,
            "answer": answers,
            "weight": weights,
            "scene_id": scene_id_list, # what is this used for?
            "question_id": ann["id"],
        }

        return out

    def __len__(self):
        return len(self.annotation)
    
    def collater(self, samples):
        pc_feat_list, points_list = [], []
        question_list, answer_list, weight_list = [], [], []
        num_answers = []

        for sample in samples:
            # stack scenes: [S, 5000, 1408]
            pc_feat_list.append(torch.stack(sample["pc_feat"], dim=0))
            points_list.append(torch.stack(sample["pc"], dim=0))
            question_list.append(sample["text_input"])
            weight_list.extend(sample["weight"])
            answer_list.extend(sample["answer"])
            num_answers.append(len(sample["answer"]))

        return {
            "pc_feat": torch.stack(pc_feat_list, dim=0),  # [B, S, 5000, 1408]
            "pc": torch.stack(points_list, dim=0),        # [B, S, 5000, 3]
            "text_input": question_list,
            "answer": answer_list,
            "weight": torch.Tensor(weight_list),
            "n_answers": torch.LongTensor(num_answers),
        }

class LongtermCaptionEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.pc_feat_root = "/local1/leisongao/data/3dllm/rgb_features"  
        self.voxel_root = "/local1/leisongao/data/3dllm/rgb_features" # default point clouds

        self.scene_ids = {}
        n = 0
        new_annotation = []
        print(f"\nEval dataset there are {len(self.annotation)} possible examples\n\n")
        for ann in self.annotation:
            try:
                valid_ann = True
                for img_id in ann["video"]:
                    img_id = img_id[5:] # only get the last video
                    if img_id not in self.scene_ids.keys():
                        self.scene_ids[img_id] = n
                        n += 1
                    if not (os.path.exists(os.path.join(self.pc_feat_root, img_id, "pcd_pos_29.pt"))):
                        valid_ann = False
                if valid_ann:
                    new_annotation.append(ann)
            except:
                pass

        self.annotation = new_annotation

        # self.annotation = [
        #     ann for ann in self.annotation if os.path.exists(os.path.join(self.pc_feat_root, ann["video"][-1][5:], "pcd_pos.pt"))
        # ]

        print(f"\nEval dataset there are {len(self.annotation)} vaild examples")

    def __getitem__(self, index):

        ann = self.annotation[index]

        captioning_prompt = self.text_processor(ann["conversations"][0]["value"])
        ##### ********* assumed that for captioning it is always only human --> gpt in pairs
        caption = ann["conversations"][1]["value"]

        pc_feat_list = []
        pc_list = []
        scene_id_list = []

        for scene in ann["video"]:
            scene_id = scene[5:]

            sample_id = torch.randint(40, ()).item()
            #### point cloud feature from the pcd_pos.pt file
            pc_feat = torch.tensor(
                torch.load(os.path.join(self.pc_feat_root, scene_id, f"pcd_feat_{sample_id}.pt"), weights_only=False, map_location="cpu")
            )
            # pc_feat = torch.load(os.path.join(self.pc_feat_root, scene_id, f"pcd_feat_{sample_id}.pt"), weights_only=False, map_location="cpu")
            #### point cloud itself from the pcd_pos.pt file
            pc = torch.tensor(
                torch.load(os.path.join(self.voxel_root, scene_id, f"pcd_pos_{sample_id}.pt"), weights_only=False, map_location="cpu")
            )
            # pc = torch.load(os.path.join(self.voxel_root, scene_id, f"pcd_pos_{sample_id}.pt"), weights_only=False, map_location="cpu")

            pc_feat_list.append(pc_feat)
            pc_list.append(pc)
            scene_id_list.append(scene_id)


        answer_weight = { 
            caption: 1
        }

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        out = {
            "pc_feat": torch.stack(pc_feat_list, dim=0),
            "pc": torch.stack(pc_list, dim=0),
            "text_input": captioning_prompt,
            "answer": answers,
            "weight": weights,
            "scene_id": scene_id_list, # what is this used for?
            "question_id": ann["id"],
        }

        return out
    




class LongtermQADataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.pc_feat_root = "/local1/leisongao/data/3dllm/rgb_features"  
        self.voxel_root = "/local1/leisongao/data/3dllm/rgb_features" # default point clouds

        self.scene_ids = {}
        n = 0
        new_annotation = []
        print(f"Train Dataset there are {len(self.annotation)} possible examples")
        for ann in self.annotation:
            try:
                valid_ann = True
                for img_id in ann["video"]:
                    img_id = img_id[5:]
                    if img_id not in self.scene_ids.keys():
                        self.scene_ids[img_id] = n
                        n += 1
                    if not (os.path.exists(os.path.join(self.pc_feat_root, img_id, "pcd_pos_39.pt"))):
                        valid_ann = False
                if valid_ann:
                    new_annotation.append(ann)
            except:
                pass

        self.annotation = new_annotation[:128]
        # self.annotation = []

        print(f"Train Dataset there are {len(self.annotation)} valid examples")

    def __getitem__(self, index):

        ann = self.annotation[index]

        qa_prompt = self.text_processor(ann["conversations"][0]["value"])
        ##### ********* assumed that for captioning it is always only human --> gpt in pairs
        caption = ann["conversations"][1]["value"]
        # question_id = ["id"]

        # scene_name = ann["scene_name"]

        pc_feat_list = []
        pc_list = []
        scene_id_list = []

        for scene in ann["video"]:
            scene_id = scene[5:]

            sample_id = torch.randint(40, ()).item()
            #### point cloud feature from the pcd_pos.pt file
            pc_feat = torch.tensor(
                torch.load(os.path.join(self.pc_feat_root, scene_id, f"pcd_feat_{sample_id}.pt"), weights_only=False, map_location="cpu")
            )
            #### point cloud itself from the pcd_pos.pt file
            pc = torch.tensor(
                torch.load(os.path.join(self.voxel_root, scene_id, f"pcd_pos_{sample_id}.pt"), weights_only=False, map_location="cpu")
            )

            pc_feat_list.append(pc_feat)
            pc_list.append(pc)
            scene_id_list.append(scene_id)


        answer_weight = { 
            caption: 1
        }

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        out = {
            "pc_feat": pc_feat_list,
            "pc": pc_list,
            "text_input": qa_prompt,
            "answer": answers,
            "weight": weights,
            # "scene_id": scene_name, # what is this used for?
            "question_id": ann["id"],
        }

        return out

    def __len__(self):
        return len(self.annotation)
    
    def collater(self, samples):
        pc_feat_list, points_list = [], []
        question_list, answer_list, weight_list = [], [], []
        num_answers = []

        for sample in samples:
            # stack scenes: [S, 5000, 1408]
            pc_feat_list.append(torch.stack(sample["pc_feat"], dim=0))
            points_list.append(torch.stack(sample["pc"], dim=0))
            question_list.append(sample["text_input"])
            weight_list.extend(sample["weight"])
            answer_list.extend(sample["answer"])
            num_answers.append(len(sample["answer"]))

        return {
            "pc_feat": torch.stack(pc_feat_list, dim=0),  # [B, S, 5000, 1408]
            "pc": torch.stack(points_list, dim=0),        # [B, S, 5000, 3]
            "text_input": question_list,
            "answer": answer_list,
            "weight": torch.Tensor(weight_list),
            "n_answers": torch.LongTensor(num_answers),
        }

class LongtermQAEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.pc_feat_root = "/local1/leisongao/data/3dllm/rgb_features"  
        self.voxel_root = "/local1/leisongao/data/3dllm/rgb_features" # default point clouds

        self.scene_ids = {}
        n = 0
        new_annotation = []
        bad_rooms = []
        print(f"Eval dataset there are {len(self.annotation)} possible examples\n\n")
        for full_ann in self.annotation:
            try:
                ann = full_ann["training_data"]
                valid_ann = True
                for img_id in ann["video"]:
                    img_id = img_id[5:] # only get the last video
                    if img_id not in self.scene_ids.keys():
                        self.scene_ids[img_id] = n
                        n += 1
                    if not (os.path.exists(os.path.join(self.pc_feat_root, img_id, "pcd_pos_39.pt"))):
                        valid_ann = False
                        bad_rooms.append(img_id)
                if valid_ann:
                    new_annotation.append(ann)
                    
            except:
                pass

        # self.annotation = new_annotation

        # self.annotation = [
        #     ann for ann in self.annotation if os.path.exists(os.path.join(self.pc_feat_root, ann["video"][-1][5:], "pcd_pos.pt"))
        # ]

        print(f"Eval dataset there are {len(new_annotation)} valid examples")
        print(f"the bad rooms are: {bad_rooms}")
        self.annotation=new_annotation

    def __getitem__(self, index):

        ann = self.annotation[index]

        captioning_prompt = self.text_processor(ann["conversations"][0]["value"])
        ##### ********* assumed that for captioning it is always only human --> gpt in pairs
        caption = ann["conversations"][1]["value"]

        # scene_name = ann["scene_name"]

        pc_feat_list = []
        pc_list = []
        scene_id_list = []

        for scene in ann["video"]:
            scene_id = scene[5:]

            sample_id = torch.randint(40, ()).item()
            #### point cloud feature from the pcd_pos.pt file
            pc_feat = torch.tensor(
                torch.load(os.path.join(self.pc_feat_root, scene_id, f"pcd_feat_{sample_id}.pt"), weights_only=False, map_location="cpu")
            )
            # pc_feat = torch.load(os.path.join(self.pc_feat_root, scene_id, f"pcd_feat_{sample_id}.pt"), weights_only=False, map_location="cpu")
            #### point cloud itself from the pcd_pos.pt file
            pc = torch.tensor(
                torch.load(os.path.join(self.voxel_root, scene_id, f"pcd_pos_{sample_id}.pt"), weights_only=False, map_location="cpu")
            )
            # pc = torch.load(os.path.join(self.voxel_root, scene_id, f"pcd_pos_{sample_id}.pt"), weights_only=False, map_location="cpu")

            pc_feat_list.append(pc_feat)
            pc_list.append(pc)
            scene_id_list.append(scene_id)


        answer_weight = { 
            caption: 1
        }

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        out = {
            "pc_feat": torch.stack(pc_feat_list, dim=0),
            "pc": torch.stack(pc_list, dim=0),
            "text_input": captioning_prompt,
            "answer": answers,
            "weight": weights,
            # "scene_id": scene_name, # what is this used for?
            "question_id": ann["id"],
        }

        return out