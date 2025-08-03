import sys
from torch.utils.data import Dataset, DataLoader
from os.path import join, exists, dirname

sys.path.append(dirname(__file__))
from CombinePTModel import CombinePTNet
from CombineLoader import CombineLoader
from CombineTrainer import CombineTrainer


class Registers:
    def __init__(self, model_type: str, data_type: str, num_parts: int = 14, body_model_type: str = 'PT'):
        self.model_type = model_type
        self.data_type = data_type
        self.num_parts = num_parts
        self.body_model_type = body_model_type

    def model_builder(self, part_weighting=True):
        if self.model_type == 'PoseCorr':
            return CombinePTNet(num_classes=3, num_parts=self.num_parts, part_weighting=part_weighting)
        elif self.model_type == 'BackGeo':
            if self.data_type == 'single':
                return CombinePTNet(num_classes=3, num_parts=self.num_parts, part_weighting=part_weighting)
            else:
                raise "BackGeo training only supports single-sided point cloud types."
        else:
            raise "Invalid model type"

    def dataloader_builder(self, data_mode, num_sampling: int = None, augment: bool = False, data_root: str = None,
                           batch_size: int = 16, num_workers: int = 8):
        if self.model_type in ['PoseCorr', 'BackGeo']:
            if self.data_type == 'single':
                data_suffix = f"tpose_corr_single"
            elif self.data_type == 'full':
                data_suffix = f"tpose_corr_full"
            else:
                raise ValueError("Data type is not valid.")
        else:
            raise "Model type is not valid."

        data_dir = f"{data_root}/{data_suffix}"
        assert exists(data_dir), f"Data set does not exist, {data_dir}"
        split_file = f"{data_dir}/split_data.npz"
        assert exists(split_file), f"Split file does not exist, {split_file}"

        dataset = CombineLoader(data_mode=data_mode, model_type=self.model_type, split_file=split_file,
                                num_sampling=num_sampling,
                                augment=True if self.model_type == 'PoseCorr' and augment else False,
                                data_root=data_dir)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(data_mode == 'train'), num_workers=num_workers)

    def trainer_builder(self, model, device, epochs, train_loader, val_loader, lr, exp_name, exp_path, optimizer, root):
        return CombineTrainer(model=model, model_type=self.model_type, device=device, epochs=epochs,
                              train_loader=train_loader, val_loader=val_loader, lr=lr, exp_name=exp_name,
                              exp_path=exp_path, optimizer=optimizer, root=root, model_attribute=self.body_model_type)

    def exp_name_builder(self, exp_suffix, part_weighting) -> str:
        weighting_suffix = 'weighting' if part_weighting else 'no_weighting'
        if self.model_type in ['PoseCorr', 'BackGeo']:
            if exp_suffix and exp_suffix is not None:
                exp_name = f"{self.model_type}_{self.data_type}_{weighting_suffix}_{exp_suffix}"
            else:
                exp_name = f"{self.model_type}_{self.data_type}_{weighting_suffix}"
        else:
            raise "Model type is not valid."
        return exp_name
