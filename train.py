# ============================================================
#  Project    : PoseNorm_PCN
#  File       : train.py
#  Author     : Ming Li
#  Copyright  : (c) 2025 by Ming Li. All rights reserved.
#  Email      : helloming@shu.edu.cn
#  License    : For academic and research use only.
#  Description: Shanghai University
# ============================================================

import os
from os.path import join, exists, dirname
import torch

from pathlib import Path
from models.Builder import Registers
from config.load_config import load_config


def main(model_type: str):
    assert model_type in ['PoseCorr', 'BackGeo'], 'The type of the trainer is not legal'

    '''Analyze parameters'''
    config, project_root = load_config()
    data_root = join(project_root, 'data')
    default_result_root = join(project_root, 'results')
    os.makedirs(default_result_root, exist_ok=True)
    base_param = config.get('base_param', {})
    model_type_train_param = config.get(model_type, {}).get('train', {})
    print(f"Parameters: base param - {base_param}\n train param - {model_type_train_param}")

    # base param
    num_parts = int(base_param.get('num_parts', 14))
    num_points = int(base_param.get('num_points', 10000))
    epochs = int(base_param.get('epochs', 800))
    num_workers = int(base_param.get('num_workers', 10))
    batch_size = int(base_param.get('batch_size', 10))
    lr = float(base_param.get('lr', 1e-3))
    optimizer = base_param.get('optimizer', 'Adam')
    device = base_param.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    out_path = Path(base_param.get('out_path', default_result_root))

    # train param
    if not model_type_train_param:
        raise "Training parameters are empty."
    if model_type == 'PoseCorr':
        data_type = model_type_train_param.get('data_type', 'single')
    elif model_type == 'BackGeo':
        data_type = 'single'
    else:
        data_type = 'full'
    part_weighting = bool(model_type_train_param.get('part_weighting', True))
    exp_suffix = '' if model_type_train_param.get('exp_suffix', None) is None else model_type_train_param.get(
        'exp_suffix', None)
    model_attribute = 'PT'

    print(f"Start training: {model_type} - {'weighting' if part_weighting else 'no_weighting'} - {data_type}")

    try:
        register = Registers(model_type=model_type, data_type=data_type, num_parts=num_parts,
                             body_model_type=model_attribute)
        exp_name = register.exp_name_builder(exp_suffix=exp_suffix, part_weighting=part_weighting)
        model = register.model_builder(part_weighting=part_weighting)
        train_loader = register.dataloader_builder(data_mode='train', num_sampling=num_points, augment=True,
                                                   data_root=data_root, batch_size=batch_size,
                                                   num_workers=num_workers)
        val_loader = register.dataloader_builder(data_mode='val', num_sampling=num_points, augment=False,
                                                 data_root=data_root, batch_size=batch_size,
                                                 num_workers=num_workers)
        trainer = register.trainer_builder(model=model, device=device, epochs=epochs, train_loader=train_loader,
                                           val_loader=val_loader, lr=lr, exp_name=exp_name, exp_path=project_root,
                                           optimizer=optimizer, root=project_root)
        trainer.combine_train()
    except ValueError as e:
        raise e


if __name__ == "__main__":
    main(model_type='PoseCorr')  # choices=['PoseCorr', 'BackGeo']
