# FILE: model_interface_prog.py

import os
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

# 导入基类
from models.model_interface_base import ModelInterfaceBase

# 导入此任务所需的特定库
from utils.utils import NLLSurvLoss, cox_log_rank
from sksurv.metrics import concordance_index_censored
from sklearn.utils import resample
import torch


class ModelInterfaceProg(ModelInterfaceBase):
    """
    用于生存分析 (Prognosis) 的模型接口.
    继承自 ModelInterfaceBase，并实现了生存分析任务特有的逻辑.
    """

    def __init__(self, model, loss, optimizer, **kargs):
        # 1. 调用父类的构造函数，处理通用初始化
        super().__init__(model, optimizer, **kargs)

        # 2. 保存此子类特有的超参数
        self.save_hyperparameters('loss')

        # 3. 定义子类特有的属性
        self.loss = NLLSurvLoss(loss.alpha_surv)
        self.shuffle = kargs['data'].data_shuffle

    def training_step(self, batch, batch_idx):
        if len(batch) == 4:
            data_WSI, label, event_time, c = batch
            results_dict = self.model(data=data_WSI)
        else:
            sample, survival = batch
            label = survival[0][0].unsqueeze(0)
            event_time = survival[0][1].unsqueeze(0)
            c = survival[0][2].unsqueeze(0)
            results_dict = self.model(data=sample)

        hazards = results_dict['hazards']
        S = results_dict['S']

        loss = self.loss(hazards=hazards, S=S, Y=label.long(), c=c)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            data_WSI, label, event_time, c = batch
            results_dict = self.model(data=data_WSI)
        else:
            sample, survival = batch
            label = survival[0][0].unsqueeze(0)
            event_time = survival[0][1].unsqueeze(0)
            c = survival[0][2].unsqueeze(0)
            results_dict = self.model(data=sample)

        hazards = results_dict['hazards']
        S = results_dict['S']

        loss = self.loss(hazards=hazards, S=S, Y=label.long(), c=c)
        risk = -torch.sum(S, dim=1).cpu().item()
        return {'loss': loss.item(), 'risk': risk, 'censorship': c.item(), 'event_time': event_time.item()}

    def validation_epoch_end(self, val_step_outputs):
        all_val_loss = np.stack([x['loss'] for x in val_step_outputs])
        all_risk_scores = np.stack([x['risk'] for x in val_step_outputs])
        all_censorships = np.stack([x['censorship'] for x in val_step_outputs])
        all_event_times = np.stack([x['event_time'] for x in val_step_outputs])

        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores,
                                             tied_tol=1e-08)[0]
        pvalue_pred = cox_log_rank(all_risk_scores, (1 - all_censorships), all_event_times)

        self.log('val_loss', np.mean(all_val_loss), prog_bar=True, on_epoch=True, logger=True)
        self.log('c_index', c_index, prog_bar=True, on_epoch=True, logger=True)
        self.log('p_value', pvalue_pred, prog_bar=True, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        if len(batch) == 4:
            data_WSI, label, event_time, c = batch
            results_dict = self.model(data=data_WSI)
        else:
            sample, survival = batch
            label = survival[0][0].unsqueeze(0)
            event_time = survival[0][1].unsqueeze(0)
            c = survival[0][2].unsqueeze(0)
            results_dict = self.model(data=sample)

        S = results_dict['S']
        risk = -torch.sum(S, dim=1).cpu().item()
        return {'risk': risk, 'censorship': c.item(), 'event_time': event_time.item(), 'S': S.detach().cpu().numpy()}

    def test_epoch_end(self, output_results):
        all_risk_scores = np.stack([x['risk'] for x in output_results])
        all_censorships = np.stack([x['censorship'] for x in output_results])
        all_event_times = np.stack([x['event_time'] for x in output_results])

        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores,
                                             tied_tol=1e-08)[0]
        pvalue_pred = cox_log_rank(all_risk_scores, (1 - all_censorships), all_event_times)
        print(f'Test C-Index: {c_index:.4f}, P-Value: {pvalue_pred:.4f}')

        # Bootstrap
        n_boots = 1000
        boot_c_indices = []
        for i in tqdm(range(n_boots), desc="Bootstrapping C-Index"):
            boot_ids = resample(np.arange(len(all_risk_scores)), replace=True)
            try:
                c_index_boot = concordance_index_censored(
                    (1 - all_censorships[boot_ids]).astype(bool),
                    all_event_times[boot_ids],
                    all_risk_scores[boot_ids],
                    tied_tol=1e-08
                )[0]
                boot_c_indices.append(c_index_boot)
            except ZeroDivisionError:
                continue

        # 计算置信区间
        ci_lower, ci_upper = np.percentile(boot_c_indices, [2.5, 97.5])
        print(f"95% CI for C-Index: [{ci_lower:.4f}, {ci_upper:.4f}]")

        # 保存结果
        results_dict = {
            'c_index': c_index,
            'c_index_low': ci_lower,
            'c_index_high': ci_upper,
            'p_value': pvalue_pred
        }
        pd.DataFrame(list(results_dict.items()), columns=['Metric', 'Value']).to_csv(
            os.path.join(self.log_path, 'test_results.csv'), index=False)