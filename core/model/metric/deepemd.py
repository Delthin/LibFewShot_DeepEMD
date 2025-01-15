import torch
import torch.nn.functional as F
from torch import nn
from .metric_model import MetricModel
from core.utils import accuracy
from qpth.qp import QPFunction
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor



class EMDSolver(nn.Module):
    """EMD solver基类"""

    def __init__(self):
        super().__init__()

    def forward(self, distance_matrix, weight1, weight2):
        raise NotImplementedError


class OpenCVSolver(EMDSolver):
    """OpenCV EMD solver实现"""

    def __init__(self, num_workers=4, chunk_size=256):
        super().__init__()
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)
        self.cpu_buffers = None

    def forward(self, distance_matrix, weight1, weight2):
        """并行处理多个EMD问题
        Args:
            distance_matrix: [batch_size, 1, N, N]
            weight1: [batch_size, 1, N]
            weight2: [batch_size, 1, N]
        Returns:
            torch.Tensor: [batch_size, N, N] flow矩阵
        """
        batch_size = distance_matrix.shape[0]
        N = distance_matrix.shape[-1]

        # 去除扩展的维度并预处理数据
        distance_matrix = distance_matrix.squeeze(1)  # [batch_size, N, N]
        weight1 = weight1.squeeze(1)  # [batch_size, N]
        weight2 = weight2.squeeze(1)  # [batch_size, N]

        # 在GPU上进行权重预处理
        weight1 = F.relu(weight1) + 1e-5
        weight2 = F.relu(weight2) + 1e-5

        weight1 = weight1 * (N / weight1.sum(dim=1, keepdim=True))
        weight2 = weight2 * (N / weight2.sum(dim=1, keepdim=True))

        # 转换为NumPy数组，一次性完成数据传输
        distance_matrix_np = distance_matrix.detach().cpu().numpy()
        weight1_np = weight1.detach().cpu().numpy()
        weight2_np = weight2.detach().cpu().numpy()

        # 准备输出数组
        flows = np.zeros((batch_size, N, N), dtype=np.float32)

        # 将数据分成小批次并行处理
        futures = []
        for start_idx in range(0, batch_size, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, batch_size)
            future = self.thread_pool.submit(
                self._process_chunk,
                start_idx,
                end_idx,
                distance_matrix_np[start_idx:end_idx],
                weight1_np[start_idx:end_idx],
                weight2_np[start_idx:end_idx],
                flows[start_idx:end_idx]
            )
            futures.append(future)

        # 等待所有计算完成
        for future in futures:
            future.result()

        # 一次性将结果转回GPU
        return torch.from_numpy(flows).cuda()

    def _process_chunk(self, start_idx, end_idx, cost_matrices, weights1, weights2, out_flows):
        """处理一个批次的EMD计算"""
        for i in range(end_idx - start_idx):
            weight1 = weights1[i].reshape(-1, 1)
            weight2 = weights2[i].reshape(-1, 1)
            cost_matrix = cost_matrices[i]

            try:
                _, _, flow = cv2.EMD(
                    weight1,
                    weight2,
                    cv2.DIST_USER,
                    cost_matrix
                )
                out_flows[i] = flow
            except Exception as e:
                print(f"Error processing EMD at index {start_idx + i}: {e}")
                # 发生错误时使用零矩阵作为fallback
                out_flows[i] = np.zeros_like(cost_matrix)

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown()


def split_batch(tensor, chunk_size):
    """将batch切分成更小的chunks"""
    return torch.split(tensor, chunk_size)


class QPTHSolver(EMDSolver):
    """QPTH EMD solver实现"""

    def __init__(self, form='QP', l2_strength=0.0001):
        super().__init__()
        self.form = form
        self.l2_strength = l2_strength

    def forward(self, distance_matrix, weight1, weight2):
        """批量处理EMD问题
        Args:
            distance_matrix: [batch_size, 1, N, N]  # [750, 1, 14, 14]
            weight1: [batch_size, 1, N]  # [750, 1, 14]
            weight2: [batch_size, 1, N]  # [750, 1, 14]
        Returns:
            torch.Tensor: [batch_size, N, N] flow矩阵
        """
        batch_size = distance_matrix.shape[0]
        N = distance_matrix.shape[2]

        # 去除扩展的维度
        distance_matrix = distance_matrix.squeeze(1)  # [batch_size, N, N]
        weight1 = weight1.squeeze(1)  # [batch_size, N]
        weight2 = weight2.squeeze(1)  # [batch_size, N]

        # 权重归一化
        weight1 = (weight1 * N) / weight1.sum(1, keepdim=True)
        weight2 = (weight2 * N) / weight2.sum(1, keepdim=True)

        # 构建QP问题
        Q, p = self._build_qp_input(distance_matrix, batch_size, N)
        flow = self._qp_solve(Q, p, weight1, weight2)

        return flow.view(batch_size, N, N)

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
        """计算EMD距离 - 并行版本
        Args:
            similarity_map: [episode_size, num_query, way_num, N, N]
            weight_1: [episode_size, num_query, way_num, N]
            weight_2: [episode_size, way_num, num_query, N]
        Returns:
            torch.Tensor: [episode_size, num_query, way_num]
        """
        # 1. 获取维度信息
        episode_size, num_query, way_num, N, _ = similarity_map.shape
        # 2. 重塑输入以进行并行计算
        similarity_map = similarity_map.view(-1, N, N)  # [episode_size*num_query*way_num, N, N]
        weight_1 = weight_1.view(-1, N)  # [episode_size*num_query*way_num, N]
        weight_2 = weight_2.transpose(1, 2).contiguous().view(-1, N)  # [episode_size*num_query*way_num, N]

        # 3. 并行计算EMD
        cost_matrix = 1 - similarity_map
        flow = self.solver(cost_matrix,
                           weight_1,
                           weight_2)  # [episode_size*num_query*way_num, N, N]

        # 4. 计算得分并重塑
        scores = (flow * similarity_map).sum((-1, -2))  # [episode_size*num_query*way_num]
        scores = scores.view(episode_size, num_query, way_num)  # [episode_size, num_query, way_num]

        return scores * (self.temperature / N)


class SimilarityLayer(nn.Module):
    def __init__(self, metric='cosine', norm='center'):
        super().__init__()
        self.metric = metric
        self.norm = norm
        
    def forward(self, proto, query):
        """计算特征相似度 - 并行版本
        Args:
            proto: [episode_size, way_num, C, 1, N]
            query: [episode_size, query_num*way_num, C, 1, N]
        Returns:
            torch.Tensor: [episode_size, num_query, way_num, N, N]
        """
        # 1. 获取维度信息
        episode_size, way_num, C, _, N = proto.shape
        num_query = query.shape[1]

        # 2. 去除多余维度并重塑
        proto = proto.squeeze(3)  # [episode_size, way_num, C, N]
        query = query.squeeze(3)  # [episode_size, query_num*way_num, C, N]

        # 3. 重塑维度用于并行计算
        proto = proto.unsqueeze(2).expand(-1, -1, num_query, -1, -1)  # [episode_size, way_num, num_query, C, N]
        query = query.view(episode_size, num_query, 1, C, N).expand(-1, -1, way_num, -1,
                                                                    -1)  # [episode_size, num_query, way_num, C, N]

        # 4. 计算相似度
        if self.metric == 'cosine':
            proto = proto.permute(0, 2, 1, 4, 3)  # [episode_size, num_query, way_num, N, C]
            query = query.permute(0, 1, 2, 4, 3)  # [episode_size, num_query, way_num, N, C]
            similarity_map = torch.matmul(proto, query.transpose(-1, -2))  # [episode_size, num_query, way_num, N, N]
            norm_proto = torch.norm(proto, p=2, dim=-1, keepdim=True)
            norm_query = torch.norm(query, p=2, dim=-1, keepdim=True)
            similarity_map = similarity_map / (norm_proto * norm_query.transpose(-1, -2) + 1e-8)

        elif self.metric == 'l2':
            proto = proto.unsqueeze(4)  # [episode_size, num_query, way_num, N, 1, C]
            query = query.unsqueeze(3)  # [episode_size, num_query, way_num, 1, N, C]

            similarity_map = 1 - (proto - query).pow(2).sum(-1)  # [episode_size, num_query, way_num, N, N]

        return similarity_map

    def normalize_feature(self, x):
        """特征归一化"""
        if self.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
        return x

    def get_weight_vector(self, A, B):
        """计算权重向量"""
        episode_size = A.shape[0]
        combinations = []

        for i in range(episode_size):
            cur_A = A[i]  # [num_query*way_num, C, N]
            cur_B = B[i]  # [way_num, C, N]

            M = cur_A.shape[0]
            N = cur_B.shape[0]

            cur_B = F.adaptive_avg_pool2d(cur_B, [1, 1])
            cur_B = cur_B.repeat(1, 1, cur_A.shape[2], cur_A.shape[3])

            cur_A = cur_A.unsqueeze(1)
            cur_B = cur_B.unsqueeze(0)

            cur_A = cur_A.repeat(1, N, 1, 1, 1)
            cur_B = cur_B.repeat(M, 1, 1, 1, 1)

            cur_combination = (cur_A * cur_B).sum(2)
            cur_combination = cur_combination.view(M, N, -1)
            cur_combination = F.relu(cur_combination) + 1e-3

            combinations.append(cur_combination)

        combination = torch.stack(combinations)  # [episode_size, M, N, -1]
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
        pos_h = torch.randint(0, feat_h - 1, (batch_size, self.num_patch)).cuda()
        pos_w = torch.randint(0, feat_w - 1, (batch_size, self.num_patch)).cuda()
        
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
            support_feat: [episode_size, way_num, C, 1, N]
        Returns:
            torch.Tensor: [episode_size, way_num, C, 1, N]
        """
        episode_size = support_feat.shape[0]
        protos = []
        for i in range(episode_size):
            # 1. 重塑support_feat维度以匹配源码
            cur_support = support_feat[i]  # [5, 640, 1, 14]
            cur_support = cur_support.expand(self.shot_num, -1, -1, -1, -1)  # [shot_num, 5, 640, 1, 14]
            cur_support = cur_support.transpose(0, 1)  # [5, shot_num, 640, 1, 14]
            C = cur_support.shape[2]  # 获取通道数
            N = cur_support.shape[-1]  # 获取最后一个维度
            cur_support = cur_support.reshape(self.way_num * self.shot_num, C, 1, N)  # [5*shot_num, 640, 1, 14]

            # 2. 初始化原型
            proto = cur_support.view(self.shot_num, self.way_num, self.hdim, 1, -1).mean(dim=0)  # [5, 640, 1, 14]
            proto = proto.clone().detach()
            proto = nn.Parameter(proto, requires_grad=True)  # 设置requires_grad=True

            # 3. 优化器设置
            optimizer = torch.optim.SGD([proto], lr=self.lr, momentum=0.9, dampening=0.9,
                                        weight_decay=self.weight_decay)

            # 4. 训练标签
            label_shot = torch.arange(self.way_num).repeat(self.shot_num).cuda()  # [5*shot_num]

            # 5. 微调过程 - 移除proto.train()
            with torch.enable_grad():  # 确保梯度计算
                for _ in range(self.update_step):
                    rand_id = torch.randperm(self.way_num * self.shot_num).cuda()
                    for j in range(0, self.way_num * self.shot_num, self.batch_size):
                        selected_id = rand_id[j: min(j + self.batch_size, self.way_num * self.shot_num)]
                        batch_shot = cur_support.reshape(self.way_num * self.shot_num, -1, 1, N)[selected_id]
                        batch_label = label_shot[selected_id]  # [bs]

                        optimizer.zero_grad()
                        logits = self.get_logits(proto,
                                                 batch_shot)  # proto: [way_num, C, 1, N], batch_shot: [bs, C, 1, N]
                        loss = F.cross_entropy(logits, batch_label)
                        loss.backward()
                        optimizer.step()
            protos.append(proto.detach())

        # 6. 恢复维度
        return torch.stack(protos)  # [episode_size, way_num, C, 1, N]

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
        return logits.squeeze(0)  # [bs, way_num]


class DeepEMD(MetricModel):
    def __init__(self, hdim, temperature=12.5, **kwargs):
        super().__init__(**kwargs)
        self.hdim = hdim
        self.temperature = temperature

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

    def _reorder_samples(self, data):
        """公共的重排函数"""
        reordered_idx = []
        samples_per_class = self.shot_num + self.query_num
        for i in range(samples_per_class):
            for j in range(self.way_num):
                idx = j * samples_per_class + i
                reordered_idx.append(idx)
        return data[reordered_idx]

    def _split_by_episode(self, features, mode):
        episode_size = features.size(0) // (self.way_num * (self.shot_num + self.query_num))
        
        # 生成正确顺序的标签
        labels_per_episode = []
        for i in range(self.shot_num + self.query_num):
            for j in range(self.way_num):
                labels_per_episode.append(j)
        
        local_labels = torch.tensor(labels_per_episode, device=self.device)
        local_labels = local_labels.repeat(episode_size)
        
        if mode == 2:
            b, c, h, w = features.shape
            
            # 关键修改：正确重组特征
            features = features.view(episode_size, -1, c, h, w)  # [episode_size, way*(shot+query), c, h, w]
            
            # 分离support和query特征
            support_features = features[:, :self.way_num*self.shot_num]
            query_features = features[:, self.way_num*self.shot_num:]
            
            # Reshape support特征为[episode_size, way_num * shot_num, c, h, w]
            support_features = support_features.contiguous()
            
            # Reshape query特征为[episode_size, way_num * query_num, c, h, w]
            query_features = query_features.contiguous()
            
            # 分离标签
            local_labels = local_labels.view(episode_size, -1)
            support_target = local_labels[:, :self.way_num*self.shot_num]
            query_target = local_labels[:, self.way_num*self.shot_num:]
            
        return support_features, query_features, support_target, query_target

    def set_forward(self, batch):
        image = batch[0]
        image = image.to(self.device)
        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))
        
        # 重排数据
        image = self._reorder_samples(image)
        
        feat = self.emb_func(image)
        if len(feat.shape) == 2:
            B, C = feat.shape
            feat = feat.view(B, C, 1, 1)
        elif len(feat.shape) == 3:
            B, C, N = feat.shape
            feat = feat.view(B, C, int(N ** 0.5), int(N ** 0.5))

        feat = self.feature_extractor(feat)
        support_feat, query_feat, support_target, query_target = self._split_by_episode(feat, mode=2)
        
        # 2. SFC微调(仅在测试时使用)
        if not self.training and self.shot_num > 1:
            support_feat = self.sfc_layer(support_feat)

        # 3. EMD计算
        similarity_map = self.similarity_layer(support_feat, query_feat)
        weight_1 = self.similarity_layer.get_weight_vector(query_feat, support_feat)
        weight_2 = self.similarity_layer.get_weight_vector(support_feat, query_feat)
        logits = self.emd_layer(similarity_map, weight_1, weight_2)

        # 4. 重塑输出
        logits = logits.reshape(episode_size, self.way_num * self.query_num, self.way_num)
        logits = logits.reshape(-1, self.way_num)  # 展平所有episode的结果
        acc = accuracy(logits, query_target.reshape(-1))
        return logits, acc

    def set_forward_loss(self, batch):
        image = batch[0]
        image = image.to(self.device)
        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))

        image = self._reorder_samples(image)
        
        feat = self.emb_func(image)
        if len(feat.shape) == 2:
            B, C = feat.shape
            feat = feat.view(B, C, 1, 1)
        elif len(feat.shape) == 3:
            B, C, N = feat.shape
            feat = feat.view(B, C, int(N ** 0.5), int(N ** 0.5))

        feat = self.feature_extractor(feat)
        support_feat, query_feat, support_target, query_target = self._split_by_episode(feat, mode=2)

        # EMD计算
        similarity_map = self.similarity_layer(support_feat, query_feat)
        
        weight_1 = self.similarity_layer.get_weight_vector(query_feat, support_feat)
        weight_2 = self.similarity_layer.get_weight_vector(support_feat, query_feat)
        
        logits = self.emd_layer(similarity_map, weight_1, weight_2)
        logits = logits.reshape(episode_size, self.way_num * self.query_num, self.way_num)
        logits = logits.reshape(-1, self.way_num)
        loss = self.loss_func(logits, query_target.reshape(-1))
        acc = accuracy(logits, query_target.reshape(-1))

        return logits, acc, loss