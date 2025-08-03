import codecs
import numpy as np
from torch.utils.data import Dataset
import torch
import random
import os
from os.path import join, split, exists, splitext, basename
import multiprocessing
from typing import Union, Tuple, List, Dict, Any, Optional
from collections.abc import Mapping, Sequence


class BaseLoader(Dataset):
    def __init__(self, mode: str, split_file: str, batch_size: int, num_workers: int, load_keys: List[str]):
        # init params
        self.mode = mode
        self.split_file = split_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_keys = load_keys
        self.valid_data = []
        self._check_params_valid()  # check param

        self.transforms = {key: None for key in self.load_keys}
        self.global_transform = None

    def _check_params_valid(self):
        errors = []
        # check data mode
        if self.mode not in ["train", "val"]:
            errors.append(f"The data mode is not legal: {self.mode}")

        # check split file
        if not exists(self.split_file):
            errors.append(f"Split file does not exist: {self.split_file}")
        else:
            try:
                raw_data = np.load(self.split_file, allow_pickle=True).item()
                if self.mode not in raw_data:
                    errors.append(f"Split file does not contain '{self.mode}' data")
                else:
                    data_all = raw_data[self.mode]
                    data_root = raw_data.get("root", "")
                    for data in data_all:
                        path = join(data_root, data)
                        if exists(path):
                            self.valid_data.append(path)
                    if len(self.valid_data) == 0:
                        errors.append(f"No valid data found for mode '{self.mode}'")
            except Exception as e:
                errors.append(f"Error loading split file: {str(e)}")

        # check batch size
        if self.batch_size <= 0:
            self.batch_size = 8
            print(f"Warning: Batch size <= 0, set to default: 8")

        # check num workers
        if self.num_workers <= 0:
            max_workers = multiprocessing.cpu_count()
            self.num_workers = max(1, int(max_workers * 0.6))  # 至少1个worker
            print(f"Warning: num_workers <= 0, set to {self.num_workers} (60% of CPU cores)")

        if errors:
            raise ValueError("\n".join(errors))

        print(f"Data loader initialized successfully. Found {len(self.valid_data)} valid samples.")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset with size {len(self)}")
        data_path = self.valid_data[index]
        result = {}
        try:
            data = np.load(data_path, allow_pickle=True).item()
            for key in self.load_keys:
                if key in data:
                    value = data[key]

                    if self.transforms[key] is not None:
                        value = self.transforms[key](value)

                    result[key] = value
                else:
                    result[key] = None
                    if self.verbose:
                        print(f"warning: key '{key}' does not exist")
            if self.global_transform is not None:
                result = self.global_transform(result)
        except Exception as e:
            print(f"Load error:  {str(e)}")
        return result

    def __len__(self):
        return len(self.valid_data)

    def get_dataloader(self, shuffle: Optional[bool] = None, **kwargs) -> torch.utils.data.DataLoader:
        """获取配置好的DataLoader"""
        if shuffle is None:
            shuffle = (self.mode == 'train')

        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
            worker_init_fn=self._worker_init_fn,
            **kwargs
        )

    @staticmethod
    def _worker_init_fn(worker_id: int):
        """工作线程初始化函数"""
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """自定义批处理函数"""
        if not batch:
            return {}

        # 获取所有可能的键
        all_keys = set()
        for sample in batch:
            all_keys.update(sample.keys())

        # 为每个键创建批处理
        collated = {}
        for key in all_keys:
            items = [sample.get(key) for sample in batch]

            # 处理None值
            if any(x is None for x in items):
                collated[key] = None
                continue

            # 处理张量
            if isinstance(items[0], torch.Tensor):
                try:
                    collated[key] = torch.stack(items)
                except RuntimeError:  # 如果尺寸不匹配
                    collated[key] = items
            # 处理numpy数组
            elif isinstance(items[0], np.ndarray):
                try:
                    collated[key] = torch.stack([torch.from_numpy(x) for x in items])
                except RuntimeError:
                    collated[key] = items
            # 处理标量
            elif isinstance(items[0], (int, float)):
                collated[key] = torch.tensor(items)
            # 其他类型保持原样
            else:
                collated[key] = items

        return collated
