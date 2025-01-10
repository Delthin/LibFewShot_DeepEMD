import torch
import torch.nn.functional as F 
from torch import nn
from .metric_model import MetricModel
from core.utils import accuracy

class ProtoLayer(nn.Module):
    """原型计算的基类"""
    def __init__(self):
        super().__init__()
    
    def forward(self, query_feat, support_feat, way_num, shot_num, query_num):
        """
        Args:
            query_feat: [t, wq, c]
            support_feat: [t, ws, c]
        """
        t, wq, c = query_feat.size()
        _, ws, _ = support_feat.size()

        # 计算原型
        query_feat = query_feat.reshape(t, way_num * query_num, c)
        support_feat = support_feat.reshape(t, way_num, shot_num, c)
        proto_feat = torch.mean(support_feat, dim=2)  # [t, way_num, c]

        return query_feat, proto_feat


class EMDLayer(ProtoLayer):
    def __init__(self, solver='opencv', form='QP', l2_strength=0.0001):
        super().__init__()
        self.solver = solver
        self.form = form
        self.l2_strength = l2_strength
    
    def forward(self, query_feat, support_feat, way_num, shot_num, query_num):
        # 获取原型表示
        query_feat, proto_feat = super().forward(query_feat, support_feat, way_num, shot_num, query_num)
        
        # EMD计算
        similarity_map = self.get_similiarity_map(query_feat, proto_feat)
        weight_1, weight_2 = self.get_weight_vector(query_feat, proto_feat)
        
        if self.solver == 'opencv':
            logits = self.emd_inference_opencv(similarity_map, weight_1, weight_2)
        else:
            logits = self.emd_inference_qpth(similarity_map, weight_1, weight_2)
            
        return logits
    
    def emd_inference_qpth(self, distance_matrix, weight1, weight2):
        """QP求解器实现"""
        # ...原emd_inference_qpth实现...
    
    def emd_inference_opencv(self, cost_matrix, weight1, weight2): 
        """OpenCV求解器实现"""
        # ...原emd_inference_opencv实现...


class FeatureExtractor(nn.Module):
    def __init__(self, feature_mode='fcn', feature_pyramid=None, patch_list='2,3', num_patch=25):
        super().__init__() 
        self.feature_mode = feature_mode
        self.feature_pyramid = [int(s) for s in feature_pyramid.split(',')] if feature_pyramid else None
        self.patch_list = [int(s) for s in patch_list.split(',')]
        self.num_patch = num_patch

    def forward(self, x):
        if self.feature_mode == 'fcn':
            return self.feature_pyramid_transform(x)
        elif self.feature_mode == 'grid':
            return self.grid_transform(x)
        return self.sampling_transform(x)
    
    def feature_pyramid_transform(self, x):
        # ...原feature_pyramid实现...
        pass
    
    def grid_transform(self, x):
        # ...原grid采样实现...
        pass
    
    def sampling_transform(self, x):
        # ...原sampling采样实现...
        pass


class SFCLayer(nn.Module):
    def __init__(self, hdim, way_num, sfc_lr=0.1, sfc_update_step=100):
        super().__init__()
        self.fc = nn.Linear(hdim, way_num)
        self.lr = sfc_lr
        self.update_step = sfc_update_step
    
    def forward(self, support_feat, support_target):
        # ...SFC微调实现...
        pass

class SimilarityLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, query_feat, support_feat):
        query_feat = self.normalize_feature(query_feat)
        support_feat = self.normalize_feature(support_feat)
        return torch.matmul(support_feat, query_feat.transpose(1,2))
    
    def normalize_feature(self, x):
        return F.normalize(x, p=2, dim=1)
    
    def get_weight_vector(self, A, B):
        #...原get_weight_vector实现...
        pass


class DeepEMD(MetricModel):
    def __init__(self, hdim, temperature=12.5, **kwargs):
        super().__init__(**kwargs)
        self.hdim = hdim
        self.temperature = temperature
        
        # 初始化各个模块
        self.feature_extractor = FeatureExtractor(**kwargs)
        self.emd_layer = EMDLayer(**kwargs)
        self.similarity_layer = SimilarityLayer()
        self.sfc_layer = SFCLayer(hdim, self.way_num, **kwargs)
        
        self.loss_func = nn.CrossEntropyLoss()

    
    def set_forward(self, batch):
        """推理阶段"""
        image, _ = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))
        
        # 1. 特征提取
        feat = self.emb_func(image)
        feat = self.feature_extractor(feat)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat)

        # 2. SFC微调(仅在测试时使用)
        if not self.training:
            support_feat = self.sfc_layer(support_feat, support_target)

        # 3. EMD计算
        similarity_map = self.similarity_layer(query_feat, support_feat)
        weight_1, weight_2 = self.similarity_layer.get_weight_vector(query_feat, support_feat)
        logits = self.emd_layer(similarity_map, weight_1, weight_2)
        
        # 4. 重塑输出
        logits = logits.reshape(episode_size * self.way_num * self.query_num, self.way_num)
        acc = accuracy(logits, query_target.reshape(-1))
        return logits, acc

    def set_forward_loss(self, batch):
        """训练阶段"""
        image, _ = batch
        image = image.to(self.device)
        
        # 1. 前向传播
        logits, acc = self.set_forward(batch) 
        
        # 2. 损失计算  
        loss = self.loss_func(logits, query_target.reshape(-1))
        
        return logits, acc, loss