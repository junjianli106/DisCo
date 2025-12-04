
import os
import numpy as np
import pandas as pd

# 导入基类
from models.model_interface_base import ModelInterfaceBase

# 导入此任务所需的特定库
from MyLoss import create_loss
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, auc as calc_auc
from sklearn.preprocessing import label_binarize
import torch


class ModelInterfaceCls(ModelInterfaceBase):
    """
    用于分类 (Classification) 的模型接口.
    继承自 ModelInterfaceBase，并实现了分类任务特有的逻辑.
    """

    def __init__(self, model, loss, optimizer, **kargs):
        # 1. 调用父类的构造函数
        super().__init__(model, optimizer, **kargs)

        # 2. 保存此子类特有的超参数
        self.save_hyperparameters('loss')

        # 3. 定义子类特有的属性
        self.loss = create_loss(loss)
        # 用于在 epoch 级别计算每个类别的准确率
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

    def _reset_acc_counter(self):
        """重置准确率计数器."""
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

    def training_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data)
        logits, Y_prob, Y_hat = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']

        loss = self.loss(logits, label)

        # 记录每个类别的准确率
        Y_hat_, Y = int(Y_hat), int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat_ == Y)

        return {'loss': loss, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label}

    def training_epoch_end(self, outputs):
        print("\n--- Training Epoch End ---")
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            acc = float(correct) / count if count > 0 else 0
            print(f'Class {c}: Accuracy {acc:.4f} ({correct}/{count})')
        self._reset_acc_counter()
        print("--------------------------\n")

    def validation_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits, Y_prob, Y_hat = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']

        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label}

    def validation_epoch_end(self, val_step_outputs):
        all_probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim=0).cpu()
        all_preds = torch.cat([x['Y_hat'] for x in val_step_outputs], dim=0).cpu()
        all_labels = torch.cat([x['label'] for x in val_step_outputs], dim=0).cpu()

        # 计算并打印每个类别的准确率
        print("\n--- Validation Epoch End ---")
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            acc = float(correct) / count if count > 0 else 0
            print(f'Class {c}: Accuracy {acc:.4f} ({correct}/{count})')
        self._reset_acc_counter()

        # 计算总体指标
        val_acc = torch.sum(all_preds == all_labels).item() / len(all_labels)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        if self.n_classes == 2:
            val_auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=list(range(self.n_classes)))
            aucs = []
            for i in range(self.n_classes):
                if len(np.unique(binary_labels[:, i])) > 1:  # 确保有两个类别
                    fpr, tpr, _ = roc_curve(binary_labels[:, i], all_probs[:, i])
                    aucs.append(calc_auc(fpr, tpr))
            val_auc = np.nanmean(aucs) if aucs else 0.0

        print(f"Overall: Acc: {val_acc:.4f}, F1 (macro): {val_f1:.4f}, AUC (macro): {val_auc:.4f}")
        print("----------------------------\n")

        self.log('val_f1', val_f1, prog_bar=True, on_epoch=True, logger=True)
        self.log('val_auc', val_auc, prog_bar=True, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        # 测试步骤的逻辑和验证步骤相同
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        print("\n--- Final Test Results ---")
        # 直接调用 validation_epoch_end 来计算和打印指标
        self.validation_epoch_end(test_step_outputs)
        # 你也可以在这里添加保存最终测试结果到文件的逻辑，类似于 validation_epoch_end