from torch import nn
from .finetuning_model import FinetuningModel
from core.utils import accuracy

# TODO
class DeepEMDPretrain(FinetuningModel):
    def __init__(self, feat_dim, num_class, **kwargs):
        super().__init__(**kwargs)
        self.feat_dim = feat_dim
        self.num_class = num_class
        
        # 预训练分类器
        self.classifier = nn.Linear(self.feat_dim, self.num_class)
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        """用于验证"""
        image, _ = batch
        feat = self.emb_func(image)
        data_shot, data_query = self.split_shot_query(feat)
        
        # k-shot情况下使用SFC
        if self.shot_num > 1:
            data_shot = self.get_sfc(data_shot)
            
        # EMD计算    
        logits = self.get_emd_distance(data_shot, data_query)
        return logits

    def set_forward_loss(self, batch):
        """用于训练"""
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)
        
        # 标准分类训练
        feat = self.emb_func(image)
        output = self.classifier(feat)
        loss = self.loss_func(output, target)
        acc = accuracy(output, target)
        
        return output, acc, loss

    def get_sfc(self, support):
        """SFC模块实现"""
        # ...类似Network.py中的实现
        pass