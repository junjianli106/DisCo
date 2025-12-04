import random
import torch
import pandas as pd
from pathlib import Path

import torch.utils.data as data
from torch.utils.data import dataloader

class ClsData(data.Dataset):
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        # ---->data和label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir)
        # ---->order
        self.shuffle = self.dataset_cfg.data_shuffle

        # special for nsclc dataset
        self.is_nsclc = False
        if 'NSCLC' in self.feature_dir:
            self.is_nsclc = True

        # ---->
        if state == 'train':
            self.data = self.slide_data['train_slide_id'].dropna().reset_index(drop=True)
            self.label = self.slide_data['train_label'].dropna().reset_index(drop=True)
        if state == 'val':
            self.data = self.slide_data['val_slide_id'].dropna().reset_index(drop=True)
            self.label = self.slide_data['val_label'].dropna().reset_index(drop=True)
        if state == 'test':
            self.data = self.slide_data['test_slide_id'].dropna().reset_index(drop=True)
            self.label = self.slide_data['test_label'].dropna().reset_index(drop=True)

        # #---->Concat related information together
        splits = [self.data, self.label]
        self.split_data = pd.concat(splits, ignore_index=True, axis=1)
        self.split_data.columns = ['slide_id', 'label']
        self.slide_id = self.split_data['slide_id'].values
        self.label = self.split_data['label'].values

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, idx):
        slide_id = self.slide_id[idx]
        label = int(self.label[idx])
        feature_dir = self.feature_dir

        if self.is_nsclc:
            cate = 'LUAD' if label == 1 else 'LUSC'
            feature_dir = feature_dir.replace('NSCLC', cate)
        try:
            full_path = Path(feature_dir) / f'{slide_id}.pt'
            features = torch.load(full_path)
            return features, label
        except Exception as e:
            print(e, slide_id)



