import torch
from torch import nn
import torch.nn.functional as F
from .finetuning_model import FinetuningModel
from core.utils import accuracy
from core.model.metric.deepemd import SFCLayer, EMDLayer, DeepEMD, FeatureExtractor


# # version1
# class DeepEMDPretrain(FinetuningModel):
#     def __init__(self, feat_dim, num_class, sfc_lr, sfc_update_step, sfc_bs, way, shot, solver, form, l2_strength, **kwargs):
#         super().__init__(**kwargs)
#         self.feat_dim = feat_dim
#         self.num_class = num_class
#
#         self.way_num = way
#         self.shot_num = shot
#         self.shot_mul_way = self.shot * self.way
#
#         self.sfc_lr = sfc_lr
#         self.sfc_update_step = sfc_update_step
#         self.sfc_bs = sfc_bs
#
#         self.solver = solver
#         self.form = form
#         self.l2_strength = l2_strength
#
#         self.deep_emd = DeepEMD()
#
#         # 预训练分类器
#         self.classifier = nn.Linear(self.feat_dim, self.num_class)
#         self.loss_func = nn.CrossEntropyLoss()
#         self.emdLayer = EMDLayer(solver=self.solver, form=self.form, l2_strength=self.l2_strength)
#         # self.split_shot_query =
#         self.sfcLayer = SFCLayer(hdim=self.feat_dim, way_num=self.way_num, sfc_lr=self.sfc_lr, sfc_update_step=self.sfc_update_step)
#
#     def set_forward(self, batch):
#         """
#         用于验证
#         Args:
#             batch: 一个batch的数据，包含image和label. image需要被划分为shot和query的部分. 前way*shot个样本为support, 后way*query个样本为query.
#         Returns:
#             logits: 预测的logits
#             acc: 预测的准确率
#         """
#         image, _ = batch
#         feat = self.emb_func(image)
#         data_shot = feat[: self.shot_mul_way] # 后加的，不知道对不对
#         data_query = feat[self.shot_mul_way:]
#         # data_shot, data_query = self.split_shot_query(feat)
#
#         # k-shot情况下使用SFC
#         if self.shot_num > 1:
#             data_shot = self.get_sfc(data_shot)
#
#         # EMD计算
#         # logits = self.get_emd_distance(data_shot, data_query)
#         logits = self.emdLayer.forward(data_query, data_shot, self.way_num, self.shot_num, len(data_query)) # 非常不确定
#         return logits, None
#
#     def set_forward_loss(self, batch):
#         """用于训练"""
#         image, target = batch
#         image = image.to(self.device)
#         target = target.to(self.device)
#
#         # 标准分类训练
#         feat = self.emb_func(image)
#         output = self.classifier(feat)
#         loss = self.loss_func(output, target)
#         acc = accuracy(output, target)
#
#         return output, acc, loss
#
#     def get_sfc(self, support):
#         """SFC模块实现"""
#         # ...类似Network.py中的实现
#         # SFC = support.view(self.shot, -1, 640, support.shape[-2], support.shape[-1]).mean(dim=0).clone().detach()
#         # SFC = nn.Parameter(SFC.detach(), requires_grad=True)
#         #
#         # optimizer = torch.optim.SGD([SFC], lr=self.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0)
#         #
#         # label_shot = torch.arange(self.way).repeat(self.shot)
#         # label_shot = label_shot.type(torch.cuda.LongTensor)
#         #
#         # with torch.enable_grad():
#         #     for k in range(0, self.sfc_update_step):
#         #         rand_id = torch.randperm(self.way * self.shot).cuda()
#         #         for j in range(0, self.way * self.shot, self.sfc_bs):
#         #             selected_id = rand_id[j: min(j + self.sfc_bs, self.way * self.shot)]
#         #             batch_shot = support[selected_id, :]
#         #             batch_label = label_shot[selected_id]
#         #             optimizer.zero_grad()
#         #             logits = self.emd_forward_1shot(SFC, batch_shot.detach())
#         #             loss = F.cross_entropy(logits, batch_label)
#         #             loss.backward()
#         #             optimizer.step()
#         # return SFC
#         return self.sfcLayer(support)

# version2
class DeepEMDPretrain(FinetuningModel):
    def __init__(self, hdim, temperature=12.5, **kwargs):
        super().__init__(**kwargs)
        self.hdim = hdim
        self.temperature = temperature
        self.deep_emd = DeepEMD(hdim=self.hdim, temperature=self.temperature, **kwargs)
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
        # 我认为这里的验证逻辑和deepemd.py的正向传播完全一致
        return self.deep_emd.set_forward(batch)

    def set_forward_loss(self, batch):
        """用于训练"""
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        # print("batch", batch[0].shape, batch[1].shape)

        # 标准分类训练
        feat = self.emb_func(image)
        # print("feat", feat.shape)
        if self.feature_mode == 'fcn':  # 最正常的训练就行，不要金字塔
            out = self.classifier(feat)
            loss = self.loss_func(out, target)
            acc = accuracy(out, target)
            return out, acc, loss
        else:
            raise NotImplementedError("Not implemented yet")



    # output = self.classifier(feat)
    # loss = self.loss_func(output, target)
    # acc = accuracy(output, target)
    # return output, acc, loss