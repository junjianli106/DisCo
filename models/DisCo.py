import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense import dense_diff_pool
from torch_geometric.utils import dense_to_sparse

import numpy as np

class GatedAttention(nn.Module):
    """Gated attention mechanism with sigmoid activation"""

    def __init__(self, input_dim=768, hidden_dim=256, n_classes=1, drop=0.25):
        super().__init__()
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(drop) if drop != 0 else nn.Identity()
        )
        self.attention_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(drop) if drop != 0 else nn.Identity()
        )
        self.attention_scorer = nn.Linear(hidden_dim, n_classes)

    def forward(self, features):
        transformed = self.feature_transform(features)
        gate = self.attention_gate(features)
        attention_weights = self.attention_scorer(transformed * gate)
        return attention_weights, features


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop != 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NystromAttention(nn.Module):
    def __init__(self, dim, dim_head, heads, num_landmarks, pinv_iterations, residual, dropout):
        super().__init__()
        print("警告：正在使用 NystromAttention 的占位符实现。")
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, heads=8):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // heads,
            heads=heads,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )

    def forward(self, x):
        is_2d_input = (x.dim() == 2)
        if is_2d_input:
            x = x.unsqueeze(0)  # 变为 (1, N, D)

        attn = self.attn(self.norm(x))
        x = x + attn

        if is_2d_input:
            return x.squeeze(0)  # 变回 (N, D)
        return x


class LocalTransPerCluster(nn.Module):
    def __init__(self, emb_dim: int, heads: int):
        super().__init__()
        self.trans = TransLayer(dim=emb_dim, heads=heads)

    def forward(self,
                patch_embeddings: torch.Tensor,
                cluster_embeddings: torch.Tensor,
                assignment_weights: torch.Tensor,
                use_soft: bool = False,
                soft_top1_threshold: float = 0.5):
        K = cluster_embeddings.size(0)
        if not use_soft:
            idx = assignment_weights.argmax(dim=1)
            groups = [torch.nonzero(idx == c, as_tuple=False).flatten() for c in range(K)]
        else:
            topw, topi = assignment_weights.max(dim=1)
            mask = topw >= soft_top1_threshold
            idx = torch.where(mask, topi, torch.full_like(topi, fill_value=-1))
            groups = [torch.nonzero(idx == c, as_tuple=False).flatten() for c in range(K)]

        patch_updated = patch_embeddings.clone()
        cluster_updated = cluster_embeddings.clone()

        for c in range(K):
            sel = groups[c]
            if sel.numel() == 0:
                continue
            p_token = cluster_embeddings[c].unsqueeze(0).unsqueeze(0)
            patches = patch_embeddings[sel].unsqueeze(0)
            x = torch.cat([p_token, patches], dim=1)
            x_out = self.trans(x)
            patches_out = x_out[:, 1:, :].squeeze(0)
            cluster_out = x_out[:, 0, :].squeeze(0)
            cluster_updated[c] = cluster_out
            patch_updated[sel] = patches_out

        return cluster_updated, patch_updated

class GNN(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, activation=nn.ReLU):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=False)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=False)
        self.act = activation()
        # --- [修改结束] ---

    def forward(self, x, adj):
        edge_index, edge_weight = dense_to_sparse(adj)

        x_out = self.act(self.conv1(x, edge_index, edge_weight))
        x_out = self.conv2(x_out, edge_index, edge_weight)

        return x_out


class LearnableClusterReducer(nn.Module):
    """通过可学习的图池化 (DiffPool) 来归约簇中心。"""

    def __init__(self, in_features: int, out_features: int, use_nonnegative_cosine_adj: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features  # K/2
        self.use_nonnegative_cosine_adj = use_nonnegative_cosine_adj

        # 嵌入GNN (学习用于计算新簇特征的节点嵌入)
        self.gnn_embed = GNN(in_features, in_features, in_features)

        # 池化GNN (学习用于计算分配矩阵 S 的节点嵌入)
        self.gnn_pool = GNN(in_features, in_features, out_features)

        # 用于存储 DiffPool 产生的辅助损失
        self.link_loss = 0.0
        self.ent_loss = 0.0

    def forward(self, x: torch.Tensor):
        num_clusters = x.shape[0]
        if num_clusters <= self.out_features:
            return x

        # 1. 动态构建图邻接矩阵 (Adjacency Matrix)
        adj = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
        if self.use_nonnegative_cosine_adj:
            adj = adj.clamp_min(0.0)
        adj.fill_diagonal_(1.0)  # 添加自环

        # 2. 通过 GNN 计算嵌入和分配矩阵
        x_embed = self.gnn_embed(x, adj)
        s_logits = self.gnn_pool(x, adj)

        # 3. 应用可微池化 (DiffPool)
        x_new, adj_new, link_loss, ent_loss = dense_diff_pool(
            x=x_embed.unsqueeze(0),  # [1, K, D]
            adj=adj.unsqueeze(0),  # [1, K, K]
            s=s_logits.unsqueeze(0),  # [1, K, K/2]
        )

        # 4. 存储辅助损失 (以供训练循环使用)
        self.link_loss = link_loss
        self.ent_loss = ent_loss

        return x_new.squeeze(0)

class DisCo(nn.Module):

    def __init__(self,
                 dim_in=768,
                 embedding_dim=512,
                 num_clusters=64,
                 n_classes=4,
                 survival=False,
                 cluster_init_path=None,
                 num_enhancers=3,
                 drop=0.25,
                 lambda_link=0.1,
                 lambda_ent=0.1,
                 hard=False,
                 similarity_method='l2',
                 use_nonnegative_cosine_adj=False,
                 use_cluster_reducer=True,
                 use_cluster_router=True,
                 use_inter_cluster_transformer=True):

        super().__init__()
        self.survival = survival
        self.hard_assignment = hard
        self.similarity_method = similarity_method
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.use_nonnegative_cosine_adj = use_nonnegative_cosine_adj
        self.use_cluster_reducer = use_cluster_reducer
        self.use_cluster_router = use_cluster_router
        self.use_inter_cluster_transformer = use_inter_cluster_transformer
        self.lambda_link = lambda_link
        self.lambda_ent = lambda_ent

        print(f"Initializing eMiCo model with:")
        print(f'  - lambda_link: {lambda_link}, lambda_ent: {lambda_ent}')
        print(f"  - use_nonnegative_cosine_adj: {self.use_nonnegative_cosine_adj}")
        print(f"  - use_cluster_reducer: {self.use_cluster_reducer} (Using Learnable DiffPool)")
        print(f"  - use_cluster_router: {self.use_cluster_router}")
        print(f"  - use_inter_cluster_transformer: {self.use_inter_cluster_transformer}")

        if cluster_init_path:
            initial_centers = torch.load(cluster_init_path)
            print('Initialize cluster centers with K-means, center shape:', initial_centers.shape)

            if isinstance(initial_centers, torch.Tensor):
                tensor_centers = initial_centers
            elif isinstance(initial_centers, np.ndarray):
                tensor_centers = torch.from_numpy(initial_centers)
            else:
                raise TypeError(f"Unsupported type for initial_centers: {type(initial_centers)}")

            tensor_centers = tensor_centers.float()


            self.cluster_centers = nn.Parameter(tensor_centers, requires_grad=True)
        else:
            self.cluster_centers = nn.Parameter(
                torch.randn(num_clusters, dim_in),
                requires_grad=True
            )

        self.patch_feature_projector = nn.Sequential(nn.Linear(dim_in, embedding_dim), nn.LeakyReLU(inplace=True))

        if self.use_cluster_reducer:
            self.dynamic_num_clusters = [num_clusters // (2 ** i) for i in range(num_enhancers + 1)]
        else:
            self.dynamic_num_clusters = [num_clusters] * (num_enhancers + 1)

        self.context_enhancers = nn.ModuleList([
            Mlp(embedding_dim, embedding_dim, embedding_dim, nn.ReLU, drop) for _ in range(num_enhancers)
        ])

        if self.use_cluster_reducer:
            self.cluster_reducers = nn.ModuleList([
                LearnableClusterReducer(
                    in_features=embedding_dim,
                    out_features=self.dynamic_num_clusters[i + 1],
                    use_nonnegative_cosine_adj=self.use_nonnegative_cosine_adj
                ) for i in range(num_enhancers)
            ])

        self.enhancer_norm_layers = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_enhancers)])
        self.cent_norm_layers = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_enhancers)])

        if self.use_inter_cluster_transformer:
            self.local_trans_layers = nn.ModuleList([
                LocalTransPerCluster(emb_dim=embedding_dim, heads=self.dynamic_num_clusters[i])
                for i in range(num_enhancers)
            ])
            self.center_layers = nn.ModuleList(
                [TransLayer(dim=embedding_dim, heads=self.dynamic_num_clusters[i]) for i in range(num_enhancers)])

        self.feature_processor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Dropout(drop) if drop != 0 else nn.Identity()
        )
        self.attention_network = GatedAttention(
            input_dim=embedding_dim, hidden_dim=embedding_dim, n_classes=1, drop=drop
        )
        self.aggregation_norm_layer = nn.LayerNorm(embedding_dim)
        self.final_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Dropout(drop) if drop > 0 else nn.Identity()
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)
        self.similarity_scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.similarity_bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _calculate_diversity_loss(self, cluster_centers):
        if cluster_centers.shape[0] <= 1:
            return 0.0
        norm_centers = F.normalize(cluster_centers, p=2, dim=1)
        similarity_matrix = torch.matmul(norm_centers, norm_centers.t())
        identity_matrix = torch.eye(norm_centers.shape[0], device=cluster_centers.device)
        loss = torch.norm(similarity_matrix - identity_matrix, p='fro')
        return loss

    def _straight_through_softmax(self, logits, hard_assignment=True, dim=-1):
        y_soft = F.softmax(logits / self.similarity_scale, dim=1)
        if hard_assignment:
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

    def _get_similarity(self, patch_embeddings, cluster_embeddings):
        if self.similarity_method == 'l2':
            return -torch.cdist(patch_embeddings, cluster_embeddings)
        else:
            return patch_embeddings @ cluster_embeddings.transpose(-2, -1)

    def _get_contextual_features(self, patch_embeddings, cluster_embeddings):
        similarity_scores = self._get_similarity(patch_embeddings, cluster_embeddings)
        assignment_weights = self._straight_through_softmax(similarity_scores, self.hard_assignment, dim=1)
        contextual_features = torch.matmul(assignment_weights, cluster_embeddings)
        return contextual_features, assignment_weights


    def forward(self, **kwargs):
        slide_features_list = kwargs['data'] if self.survival else [kwargs['data']]
        processed_slide_embeddings = []

        total_link_loss = 0.0
        total_ent_loss = 0.0

        for slide_patch_features in slide_features_list:
            patch_features = slide_patch_features.float().squeeze(0)
            patch_embeddings = self.patch_feature_projector(patch_features)
            cluster_embeddings = self.patch_feature_projector(self.cluster_centers)

            for i, enhancer_mlp in enumerate(self.context_enhancers):
                contextual_features, assignment_weights = self._get_contextual_features(
                    patch_embeddings, cluster_embeddings
                )

                if self.use_cluster_router:
                    patch_embeddings = patch_embeddings + contextual_features

                if self.use_inter_cluster_transformer:
                    cluster_embeddings, patch_embeddings = self.local_trans_layers[i](
                        patch_embeddings=patch_embeddings,
                        cluster_embeddings=cluster_embeddings,
                        assignment_weights=assignment_weights,
                        use_soft=not self.hard_assignment,
                        soft_top1_threshold=0.5
                    )

                patch_embeddings = self.enhancer_norm_layers[i](patch_embeddings)
                patch_embeddings = patch_embeddings + enhancer_mlp(patch_embeddings)

                if self.use_cluster_reducer:
                    cluster_embeddings_before_reduce = cluster_embeddings

                    reducer_layer = self.cluster_reducers[i]
                    cluster_embeddings = reducer_layer(cluster_embeddings_before_reduce)

                    total_link_loss = total_link_loss + reducer_layer.link_loss
                    total_ent_loss = total_ent_loss + reducer_layer.ent_loss

            enhanced_embeddings = torch.cat([patch_embeddings, cluster_embeddings], dim=0)
            processed_embeddings = self.feature_processor(enhanced_embeddings)
            processed_slide_embeddings.append(processed_embeddings)

        aggregated_features = torch.cat(processed_slide_embeddings, dim=0)
        aggregated_features = self.aggregation_norm_layer(aggregated_features)

        attention_scores, _ = self.attention_network(aggregated_features)
        attention_weights = F.softmax(attention_scores.transpose(0, 1), dim=1)
        slide_level_representation = torch.mm(attention_weights, aggregated_features)

        final_features = self.final_projector(slide_level_representation).squeeze()
        if final_features.dim() == 1:
            final_features = final_features.unsqueeze(0)
        logits = self.classifier(final_features)

        if self.survival:
            Y_hat = torch.argmax(logits, dim=1)
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            results_dict = {'hazards': hazards, 'S': S, 'Y_hat': Y_hat}
        else:
            Y_prob = F.softmax(logits, dim=1)
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}

        results_dict['link_loss'] = self.lambda_link * total_link_loss
        results_dict['ent_loss'] = self.lambda_ent * total_ent_loss

        return results_dict
