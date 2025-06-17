"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.longterm_datasets import LongtermCaptionDataset, LongtermCaptionEvalDataset


@registry.register_builder("longterm_caption")
class LongtermCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LongtermCaptionDataset
    eval_dataset_cls = LongtermCaptionEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/longterm_caption/defaults.yaml"}
