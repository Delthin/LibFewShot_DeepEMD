import torch
import torch.nn.functional as F 
from torch import nn
from .metric_model import MetricModel
from core.utils import accuracy
from qpth.qp import QPFunction
import cv2

class EMDSolver(nn.Module):
    """EMD solver基类"""
    def __init__(self):
        super().__init__()
    
    def forward(self, distance_matrix, weight1, weight2):
        raise NotImplementedError
        
class OpenCVSolver(EMDSolver):
    """OpenCV EMD solver实现"""
    def forward(self, distance_matrix, weight1, weight2):
        """
        Args:
            distance_matrix: [1, N, N]  # [1, 14, 14]
            weight1: [1, N]  # [1, 14]
            weight2: [1, N]  # [1, 14]
        Returns:
            torch.Tensor: [1, N, N] flow矩阵
        """
        # 去除batch维度
        distance_matrix = distance_matrix.squeeze(0)  # [N, N]
        weight1 = weight1.squeeze(0)  # [N]
        weight2 = weight2.squeeze(0)  # [N]
        
        # 计算单个EMD
        _, flow = self._solve_single(
            distance_matrix,
            weight1,
            weight2
        )
        # 添加batch维度返回
        return torch.from_numpy(flow).cuda().unsqueeze(0)  # [1, N, N]

    def _solve_single(self, cost_matrix, weight1, weight2):
        """解决单个EMD问题
        Args:
            cost_matrix: [N, N]  # [14, 14]
            weight1: [N]  # [14]
            weight2: [N]  # [14]
        """
        # 转换为float32类型
        cost_matrix = cost_matrix.detach().cpu().numpy()
        
        weight1 = F.relu(weight1) + 1e-5
        weight2 = F.relu(weight2) + 1e-5
        
        weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
        weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()
        
        cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
        return cost, flow

class QPTHSolver(EMDSolver):
    """QPTH EMD solver实现"""
    def __init__(self, form='QP', l2_strength=0.0001):
        super().__init__()
        self.form = form
        self.l2_strength = l2_strength
        
    def forward(self, distance_matrix, weight1, weight2):
        """使用QPTH计算EMD距离
        Args:
            distance_matrix (torch.Tensor): [batch_size, num_node, num_node]
            weight1 (torch.Tensor): [batch_size, num_node] 
            weight2 (torch.Tensor): [batch_size, num_node]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (emd_score, flow)
        """
        weight1 = (weight1 * weight1.shape[-1]) / weight1.sum(1).unsqueeze(1)
        weight2 = (weight2 * weight2.shape[-1]) / weight2.sum(1).unsqueeze(1)
        
        nbatch = distance_matrix.shape[0]
        nelement_distmatrix = distance_matrix.shape[1] * distance_matrix.shape[2]
        
        Q_1 = distance_matrix.view(-1, 1, nelement_distmatrix).double()
        
        if self.form == 'QP':
            Q = torch.bmm(Q_1.transpose(2, 1), Q_1).double().cuda() + 1e-4 * torch.eye(
                nelement_distmatrix).double().cuda().unsqueeze(0).repeat(nbatch, 1, 1)
            p = torch.zeros(nbatch, nelement_distmatrix).double().cuda()
        else:
            Q = (self.l2_strength * torch.eye(nelement_distmatrix).double()).cuda().unsqueeze(0).repeat(nbatch, 1, 1)
            p = distance_matrix.view(nbatch, nelement_distmatrix).double()
            
        flow = self._qp_solve(Q, p, weight1, weight2)
        return flow
        
    def _qp_solve(self, Q, p, weight1, weight2):
        """求解QP问题
        
        Args:
            Q: [batch_size, num_vars, num_vars] 二次项矩阵
            p: [batch_size, num_vars] 一次项系数
            weight1: [batch_size, num_node] 源权重
            weight2: [batch_size, num_node] 目标权重
            
        Returns:
            torch.Tensor: [batch_size, num_node, num_node] 最优流矩阵
        """
        nbatch = Q.shape[0]
        nelement_distmatrix = Q.shape[1]
        nelement_weight1 = weight1.shape[1]
        nelement_weight2 = weight2.shape[1]

        # 构建不等式约束 G*x <= h
        # 1. xij >= 0
        G_1 = -torch.eye(nelement_distmatrix).double().cuda().unsqueeze(0).repeat(nbatch, 1, 1)
        
        # 2. sum_j(xij) <= si, sum_i(xij) <= dj
        G_2 = torch.zeros([nbatch, nelement_weight1 + nelement_weight2, nelement_distmatrix]).double().cuda()
        for i in range(nelement_weight1):
            G_2[:, i, nelement_weight2 * i:nelement_weight2 * (i + 1)] = 1
        for j in range(nelement_weight2):
            G_2[:, nelement_weight1 + j, j::nelement_weight2] = 1
        
        G = torch.cat((G_1, G_2), 1)
        h_1 = torch.zeros(nbatch, nelement_distmatrix).double().cuda()
        h_2 = torch.cat([weight1, weight2], 1).double()
        h = torch.cat((h_1, h_2), 1)

        # 构建等式约束 A*x = b
        # sum_ij(x_ij) = min(sum(si), sum(dj))
        A = torch.ones(nbatch, 1, nelement_distmatrix).double().cuda()
        b = torch.min(torch.sum(weight1, 1), torch.sum(weight2, 1)).unsqueeze(1).double()

        # 求解QP问题
        flow = QPFunction(verbose=-1)(Q, p, G, h, A, b)
        
        # 重塑flow矩阵
        flow = flow.view(nbatch, nelement_weight1, nelement_weight2)
        
        return flow

class EMDLayer(nn.Module):
    """EMD计算层"""
    def __init__(self, temperature=12.5, solver='opencv', form='QP', l2_strength=0.0001):
        super().__init__()
        self.temperature = temperature
        
        if solver == 'opencv':
            self.solver = OpenCVSolver()
        else:
            self.solver = QPTHSolver(form, l2_strength)
            
    def forward(self, similarity_map, weight_1, weight_2):
        """计算EMD距离
        Args:
            similarity_map: [num_query, way_num, N, N]  # [75, 5, 14, 14]
            weight_1: [num_query, way_num, N]          # [75, 5, 14]
            weight_2: [way_num, num_query, N]          # [5, 75, 14]
        Returns:
            torch.Tensor: [num_query, way_num] EMD距离
        """
        num_query = similarity_map.shape[0]  # 75
        num_proto = similarity_map.shape[1]  # 5
        num_node = similarity_map.shape[-1]  # 14
        
        logits = torch.zeros(num_query, num_proto).to(similarity_map.device)
        
        # 遍历每个查询-原型对
        for i in range(num_query):
            for j in range(num_proto):
                # 提取单个cost矩阵和权重向量
                cost_matrix = 1 - similarity_map[i, j]  # [14, 14]
                w1 = weight_1[i, j]                     # [14]
                w2 = weight_2[j, i]                     # [14]
                
                # 调用solver计算flow
                flow = self.solver(cost_matrix.unsqueeze(0), 
                                 w1.unsqueeze(0),
                                 w2.unsqueeze(0))       # [1, 14, 14]
                
                # 计算EMD得分
                score = (flow.squeeze(0) * similarity_map[i, j]).sum()
                logits[i, j] = score
                
        return logits * (self.temperature / num_node)

class SimilarityLayer(nn.Module):
    def __init__(self, metric='cosine', norm='center'):
        super().__init__()
        self.metric = metric
        self.norm = norm
    
    def forward(self, proto, query):
        """计算特征相似度
        Args:
            proto: [episode_size, way_num, C, 1, N] -> [way_num, C, N]
            query: [episode_size, query_num*way_num, C, 1, N] -> [query_num*way_num, C, N]
        Returns:
            torch.Tensor: [num_query, way_num, N, N] 相似度矩阵
        """
        # 1. 去除多余维度
        proto = proto.squeeze(0).squeeze(2)  # [way_num, C, N]
        query = query.squeeze(0).squeeze(2)  # [query_num*way_num, C, N]
        
        way_num = proto.shape[0]
        num_query = query.shape[0]
        
        # 2. 重复扩展
        proto = proto.unsqueeze(0).repeat(num_query, 1, 1, 1)  # [num_query, way_num, C, N]
        query = query.unsqueeze(1).repeat(1, way_num, 1, 1)    # [num_query, way_num, C, N]
        # 3. 计算相似度
        if self.metric == 'cosine':
            proto = proto.permute(0, 1, 3, 2)  # [num_query, way_num, N, C]
            query = query.permute(0, 1, 3, 2)  # [num_query, way_num, N, C]
            
            proto = proto.unsqueeze(3)  # [num_query, way_num, N, 1, C]
            query = query.unsqueeze(2)  # [num_query, way_num, 1, N, C]
            
            similarity_map = F.cosine_similarity(proto, query, dim=-1)  # [num_query, way_num, N, N]
            
        elif self.metric == 'l2':
            proto = proto.permute(0, 1, 3, 2)  # [num_query, way_num, N, C]
            query = query.permute(0, 1, 3, 2)  # [num_query, way_num, N, C]
            
            proto = proto.unsqueeze(3)  # [num_query, way_num, N, 1, C]
            query = query.unsqueeze(2)  # [num_query, way_num, 1, N, C]
            
            l2_distance = (proto - query).pow(2).sum(-1)  # [num_query, way_num, N, N]
            similarity_map = 1 - l2_distance  # 转换为相似度
            
        return similarity_map
        
    def normalize_feature(self, x):
        """特征归一化"""
        if self.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
        return x
        
    def get_weight_vector(self, A, B):
        """计算权重向量
        """
        A = A.squeeze(0)
        B = B.squeeze(0)
        
        M = A.shape[0]
        N = B.shape[0]

        B = F.adaptive_avg_pool2d(B, [1, 1])
        B = B.repeat(1, 1, A.shape[2], A.shape[3])

        A = A.unsqueeze(1)
        B = B.unsqueeze(0)

        A = A.repeat(1, N, 1, 1, 1)
        B = B.repeat(M, 1, 1, 1, 1)

        combination = (A * B).sum(2)
        combination = combination.view(M, N, -1)
        combination = F.relu(combination) + 1e-3
        return combination


class FeatureExtractor(nn.Module):
    def __init__(self, patch_ratio=2.0, feature_mode='fcn', 
                feature_pyramid=None, patch_list='2,3', num_patch=25):
        super().__init__()
        self.feature_mode = feature_mode
        self.patch_ratio = patch_ratio
        self.feature_pyramid = [int(s) for s in feature_pyramid.split(',')] if feature_pyramid else None
        self.patch_list = [int(s) for s in patch_list.split(',')]
        self.num_patch = num_patch

    def forward(self, x):
        """特征提取转换
        
        Args:
            x (torch.Tensor): [batch_size, channel, height, width] 输入特征
            
        Returns:
            torch.Tensor: 转换后的特征,根据mode不同输出形状不同:
                - fcn: [batch_size, channel, 1, num_points]
                - grid: [batch_size, num_patch, channel]
                - sampling: [batch_size, num_patch, channel]
        """
        if self.feature_mode == 'fcn':
            return self.feature_pyramid_transform(x)
        elif self.feature_mode == 'grid':
            return self.grid_transform(x)
        return self.sampling_transform(x)
    
    def feature_pyramid_transform(self, x):
        """特征金字塔变换
        
        Args:
            x (torch.Tensor): [batch_size, channel, height, width] 输入特征
            
        Returns:
            torch.Tensor: [batch_size, channel, 1, total_points] 金字塔特征
        """
        feature_list = []
        # 构建多尺度特征金字塔
        for size in self.feature_pyramid:
            feature_list.append(
                F.adaptive_avg_pool2d(x, size).view(
                    x.shape[0], x.shape[1], 1, -1
                )
            )
        # 添加原始特征
        feature_list.append(x.view(x.shape[0], x.shape[1], 1, -1))
        # 拼接所有尺度特征
        out = torch.cat(feature_list, dim=-1)
        return out
    
    def grid_transform(self, x):
        """网格采样变换
        
        Args:
            x (torch.Tensor): [batch_size, channel, height, width] 输入特征
            
        Returns:
            torch.Tensor: [batch_size, num_patch, channel] 网格特征
        """
        patch_list = []
        for grid_size in self.patch_list:
            # 计算网格大小
            stride = x.shape[-1] // grid_size
            patch_size = int(stride * self.patch_ratio)
            
            # 提取网格patch
            for i in range(grid_size):
                for j in range(grid_size):
                    start_x = min(stride * i, x.shape[-2] - patch_size)
                    start_y = min(stride * j, x.shape[-1] - patch_size)
                    patch = x[:, :, 
                            start_x:start_x + patch_size,
                            start_y:start_y + patch_size]
                    patch = F.adaptive_avg_pool2d(patch, 1)
                    patch_list.append(patch)
                    
        patches = torch.cat(patch_list, dim=-1)
        patches = patches.squeeze(-2)
        patches = patches.permute(0, 2, 1)
        return patches
    
    def sampling_transform(self, x):
        """随机采样变换
        
        Args:
            x (torch.Tensor): [batch_size, channel, height, width] 输入特征
            
        Returns:
            torch.Tensor: [batch_size, num_patch, channel] 采样特征
        """
        batch_size = x.shape[0]
        
        # 随机采样位置
        feat_h, feat_w = x.shape[-2:]
        pos_h = torch.randint(0, feat_h-1, (batch_size, self.num_patch)).cuda()
        pos_w = torch.randint(0, feat_w-1, (batch_size, self.num_patch)).cuda()
        
        patches = []
        for i in range(batch_size):
            # 提取每个位置的特征
            patch = x[i, :, pos_h[i], pos_w[i]]  # [channel, num_patch]
            patches.append(patch.t())  # [num_patch, channel]
            
        patches = torch.stack(patches)
        return patches

class SFCLayer(nn.Module):
    def __init__(self, hdim, way_num, shot_num, sfc_lr=0.1, 
                 sfc_update_step=100, sfc_bs=4, sfc_wd=0):
        super().__init__()
        self.hdim = hdim
        self.way_num = way_num
        self.shot_num = shot_num
        self.lr = sfc_lr
        self.update_step = sfc_update_step 
        self.batch_size = sfc_bs
        self.weight_decay = sfc_wd
        self.similarity_layer = SimilarityLayer(metric='cosine', norm='center')
        self.emd_layer = EMDLayer(temperature=12.5, solver='opencv')

    def forward(self, support_feat):
        """SFC微调
        Args:
            support_feat: [1, 5, 640, 1, 14]  # [episode_size, way_num, C, 1, N]
        Returns:
            proto: [1, way_num, C, 1, N]  # [1, 5, 640, 1, 14]
        """
        # 1. 重塑support_feat维度以匹配源码
        support_feat = support_feat.squeeze(0)  # [5, 640, 1, 14]
        support_feat = support_feat.expand(self.shot_num, -1, -1, -1, -1)  # [shot_num, 5, 640, 1, 14]
        support_feat = support_feat.transpose(0, 1)  # [5, shot_num, 640, 1, 14]
        support_feat = support_feat.reshape(self.way_num * self.shot_num, -1, 1, 14)  # [5*shot_num, 640, 1, 14]
        
        # 2. 初始化原型
        proto = support_feat.view(self.shot_num, self.way_num, self.hdim, 1, -1).mean(dim=0)  # [5, 640, 1, 14]
        proto = proto.clone().detach()
        proto = nn.Parameter(proto, requires_grad=True)  # 设置requires_grad=True
        
        # 3. 优化器设置
        optimizer = torch.optim.SGD([proto], lr=self.lr, momentum=0.9, dampening=0.9, weight_decay=self.weight_decay)
        
        # 4. 训练标签
        label_shot = torch.arange(self.way_num).repeat(self.shot_num).cuda()  # [5*shot_num]
        
        # 5. 微调过程 - 移除proto.train()
        with torch.enable_grad():  # 确保梯度计算
            for _ in range(self.update_step):
                rand_id = torch.randperm(self.way_num * self.shot_num).cuda()
                for j in range(0, self.way_num * self.shot_num, self.batch_size):
                    selected_id = rand_id[j: min(j + self.batch_size, self.way_num * self.shot_num)]
                    batch_shot = support_feat[selected_id]  # [bs, 640, 1, 14]
                    batch_label = label_shot[selected_id]  # [bs]
                    
                    optimizer.zero_grad()
                    logits = self.get_logits(proto, batch_shot)
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        
        # 6. 恢复维度
        return proto.detach().unsqueeze(0)  # [1, 5, 640, 1, 14]

    def get_logits(self, proto, query):
        """计算分类得分
        Args:
            proto: [way_num, C, 1, N]  # [5, 640, 1, 14]
            query: [bs, C, 1, N]  # [bs, 640, 1, 14]
        Returns:
            torch.Tensor: [bs, way_num] 分类得分
        """
        # 使用已经定义好的similarity_layer
        similarity_map = self.similarity_layer(proto.unsqueeze(0), query.unsqueeze(0))
        weight_1 = self.similarity_layer.get_weight_vector(query.unsqueeze(0), proto.unsqueeze(0))
        weight_2 = self.similarity_layer.get_weight_vector(proto.unsqueeze(0), query.unsqueeze(0))
        
        # 使用已经定义好的emd_layer
        logits = self.emd_layer(similarity_map, weight_1, weight_2)
        
        return logits


class DeepEMD(MetricModel):
    def __init__(self, hdim, temperature=12.5, **kwargs):
        super().__init__(**kwargs)
        self.hdim = hdim
        self.temperature = temperature
        self.n_gpu = kwargs.get('n_gpu', 1)
        
        # 1. FeatureExtractor参数
        feature_kwargs = {
            'feature_mode': kwargs.get('feature_mode', 'fcn'),
            'patch_ratio': kwargs.get('patch_ratio', 2.0),
            'feature_pyramid': kwargs.get('feature_pyramid', None),
            'patch_list': kwargs.get('patch_list', '2,3'),
            'num_patch': kwargs.get('num_patch', 25)
        }
        
        # 2. EMDLayer参数
        emd_kwargs = {
            'temperature': temperature,
            'solver': kwargs.get('solver', 'opencv'),
            'form': kwargs.get('form', 'QP'),
            'l2_strength': kwargs.get('l2_strength', 0.0001)
        }
        
        # 3. SimilarityLayer参数 
        sim_kwargs = {
            'metric': kwargs.get('metric', 'cosine'),
            'norm': kwargs.get('norm', 'center')
        }
        
        # 4. SFCLayer参数
        sfc_kwargs = {
            'hdim': hdim,
            'way_num': self.way_num,
            'shot_num': self.shot_num,
            'sfc_lr': kwargs.get('sfc_lr', 0.1),
            'sfc_update_step': kwargs.get('sfc_update_step', 100),
            'sfc_bs': kwargs.get('sfc_bs', 4),
            'sfc_wd': kwargs.get('sfc_wd', 0)
        }
        
        # 初始化各个模块
        self.feature_extractor = FeatureExtractor(**feature_kwargs)
        self.emd_layer = EMDLayer(**emd_kwargs)
        self.similarity_layer = SimilarityLayer(**sim_kwargs)
        self.sfc_layer = SFCLayer(**sfc_kwargs)
        
        self.loss_func = nn.CrossEntropyLoss()

    
    def set_forward(self, batch):
        image, _ = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))
        
        # 1. 特征提取
        feat = self.emb_func(image) # [80, 640]
        # 修正维度
        if len(feat.shape) == 2:
            B, C = feat.shape
            feat = feat.view(B, C, 1, 1) # [80, 640, 1, 1]
        elif len(feat.shape) == 3:
            B, C, N = feat.shape 
            feat = feat.view(B, C, int(N**0.5), int(N**0.5))
            
        feat = self.feature_extractor(feat) # [80, 640, 1, 14] (fcn mode)
        if self.feature_mode == 'fcn':
            mode = 2  # 4D
        else: 
            mode = 1  # 3D
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=mode)
        # 2. SFC微调(仅在测试时使用)
        if not self.training and self.shot_num >= 1:
            support_feat = self.sfc_layer(support_feat)

        # 3. EMD计算
        similarity_map = self.similarity_layer(support_feat, query_feat)
        weight_1 = self.similarity_layer.get_weight_vector(query_feat, support_feat)
        weight_2 = self.similarity_layer.get_weight_vector(support_feat, query_feat)        
        logits = self.emd_layer(similarity_map, weight_1, weight_2)
        
        # 4. 重塑输出
        logits = logits.reshape(episode_size * self.way_num * self.query_num, self.way_num)
        acc = accuracy(logits, query_target.reshape(-1))
        return logits, acc

    def set_forward_loss(self, batch):
        image, _ = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))
        
        feat = self.emb_func(image) 

        if len(feat.shape) == 2:
            B, C = feat.shape
            feat = feat.view(B, C, 1, 1) 
        elif len(feat.shape) == 3:
            B, C, N = feat.shape 
            feat = feat.view(B, C, int(N**0.5), int(N**0.5))
        feat = self.feature_extractor(feat) 
        
        if self.feature_mode == 'fcn':
            mode = 2
        else:
            mode = 1
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=mode)
        # while training, we don't use the SFC layer
        similarity_map = self.similarity_layer(support_feat, query_feat)
        weight_1 = self.similarity_layer.get_weight_vector(query_feat, support_feat)
        weight_2 = self.similarity_layer.get_weight_vector(support_feat, query_feat)
        logits = self.emd_layer(similarity_map, weight_1, weight_2)
        
        logits = logits.reshape(episode_size * self.way_num * self.query_num, self.way_num)
        loss = self.loss_func(logits, query_target.reshape(-1))
        acc = accuracy(logits, query_target.reshape(-1))
        
        return logits, acc, loss