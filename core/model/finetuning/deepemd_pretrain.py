import torch
from torch import nn
import torch.nn.functional as F
from .finetuning_model import FinetuningModel
from core.utils import accuracy
from core.model.metric.deepemd import SFCLayer, EMDLayer, DeepEMD, FeatureExtractor

class DeepEMDPretrain(FinetuningModel):
    def __init__(self, hdim, temperature=12.5, **kwargs):
        super().__init__(**kwargs)
        self.hdim = hdim
        self.temperature = temperature
        
        self.deep_emd = DeepEMD(hdim=self.hdim, temperature=self.temperature,pretrain=True, **kwargs)
        self.classifier = nn.Linear(self.hdim, self.num_class)
        self.loss_func = nn.CrossEntropyLoss()

        feature_kwargs = {
            'feature_mode': kwargs.get('feature_mode', 'fcn'),
            'patch_ratio': kwargs.get('patch_ratio', 2.0),
            'feature_pyramid': kwargs.get('feature_pyramid', None),
            'patch_list': kwargs.get('patch_list', '2,3'),
            'num_patch': kwargs.get('num_patch', 25)
        }
        self.feature_extractor = FeatureExtractor(**feature_kwargs)
        

    def set_forward(self, batch):
        """
        用于验证
        Args:
            batch: 一个batch的数据，包含image和label. image需要被划分为shot和query的部分.
        Returns:
            logits: 预测的logits
            acc: 预测的准确率
        """
        return self.deep_emd.set_forward(batch)

    def set_forward_loss(self, batch):
        """用于训练"""
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        # 标准分类训练
        feat = self.emb_func(image)
        if self.feature_mode=='fcn': # 最正常的训练就行
            out = self.classifier(feat)
            loss = self.loss_func(out, target)
            acc = accuracy(out, target)
            return out, acc, loss
        else:
            # 特征提取器提取特征
            if len(feat.shape) == 2:
                B, C = feat.shape
                feat = feat.reshape(B, C, 1, 1)
            elif len(feat.shape) == 3:
                B, C, N = feat.shape
                feat = feat.reshape(B, C, int(N ** 0.5), int(N ** 0.5))
            feat = self.feature_extractor.forward(feat) # 特征提取器提取特征 feat: [batch_size, num_patch, hdim]
            num_patch = feat.shape[1]
            feat = feat.reshape(-1, feat.shape[-1])  # feat: [batch_size*num_patch, hdim]
            target = target.repeat(num_patch)
            out = self.classifier(feat)
            loss = self.loss_func(out, target)
            acc = accuracy(out, target)
            return out, acc, loss
            
        
