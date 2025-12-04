import os
import argparse
from pathlib import Path
import numpy as np
import glob
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import setproctitle

from datasets import DataInterface
from models import ModelInterfaceProg, ModelInterfaceCls
from glob import glob
from utils.utils import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='prog', type=str, choices=['cls', 'prog'])
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('--gpus', default=[0])
    parser.add_argument('--fold', default=2)
    return parser.parse_args()

def initialize_trainer(cfg):
    return Trainer(
        num_sanity_val_steps=0,
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs=cfg.General.epochs,
        gpus=cfg.General.gpus,
        amp_level=cfg.General.amp_level,
        precision=cfg.General.precision,
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
    )

def load_data_interface(cfg):
    return DataInterface(
        train_batch_size=cfg.Data.train_dataloader.batch_size,
        train_num_workers=cfg.Data.train_dataloader.num_workers,
        test_batch_size=cfg.Data.test_dataloader.batch_size,
        test_num_workers=cfg.Data.test_dataloader.num_workers,
        dataset_name=cfg.Data.dataset_name,
        dataset_cfg=cfg.Data,
    )

def load_model_interface(cfg):
    model_params = {
        'model': cfg.Model,
        'loss': cfg.Loss,
        'optimizer': cfg.Optimizer,
        'data': cfg.Data,
        'log': cfg.log_path
    }
    if cfg.task == 'prog':
        return ModelInterfaceProg(**model_params)
    elif cfg.task == 'cls':
        return ModelInterfaceCls(**model_params)
    else:
        raise NotImplementedError

def main(cfg):
    pl.seed_everything(cfg.General.seed)
    cfg.load_loggers = load_loggers(cfg)
    cfg.callbacks = load_callbacks(cfg)
    setproctitle.setproctitle(f'Stage: {cfg.General.server} {cfg.config} fold{cfg.Data.fold}')

    dm = load_data_interface(cfg)
    model = load_model_interface(cfg)
    trainer = initialize_trainer(cfg)

    if cfg.General.server == 'train':
        trainer.fit(model=model, datamodule=dm)
    else:
        model_paths = [str(path) for path in glob(f'{cfg.log_path}/*.ckpt') if 'epoch' in str(path)]
        for path in model_paths:
            print(path)
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':
    args = parse_arguments()
    cfg = read_yaml(args.config)

    cfg.task = args.task
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold

    if 'NSCLC' in cfg.Data.data_dir and args.task == 'cls':
        cfg.Model.cluster_init_path = cfg.Data.cluster_init_path.replace('NSCLC', 'LUAD')
    cfg.Model.cluster_init_path = cfg.Model.cluster_init_path.replace('cluster_num', str(cfg.Model.num_clusters))

    print(cfg.General.log_path)
    print(cfg.Data.data_dir)

    main(cfg)