import torch
import torch.nn as nn
import torch.nn.functional as F


class MCEdgeDropBlock2d(nn.Module):
    def __init__(
        self,
        gamma=0.03,
        block_size=5,
        lambda_edge=3.0,
        eps=1e-6,
        always_on=True
    ):
        super().__init__()
        self.gamma = gamma
        self.block_size = block_size
        self.lambda_edge = lambda_edge
        self.eps = eps
        self.always_on = always_on

        if block_size % 2 == 0:
            raise ValueError("block_size는 홀수로 설정하세요. 예: 3, 5, 7")
    
    def feature_to_edge_map(self, feature, eps=1e-6, detach=True):    
        # """
        # YOLOv5 neck에서 나온 feature map을 받아 edge map을 계산합니다.

        # feature: neck feature map [B, C, H, W]

        # 규칙:
        # 1) contrast map:
        #    contrast(i, j) = feature(i, j) - 주변 8방향 이웃 평균

        # 2) edge map:
        #    edge(i, j) = contrast(i, j) * 
        #                 contrast 부호가 같은 주변 8방향 이웃 개수

        # 반환:
        # edge map [B, C, H, W]
        # """

        if feature.dim() != 4:
            raise ValueError(
                f"feature는 [B, C, H, W] 형태여야 합니다. 현재 shape: {feature.shape}"
            )

        if detach:
            feature = feature.detach()

        B, C, H, W = feature.shape
        device = feature.device
        dtype = feature.dtype

        # -------------------------------------------------
        # 1. 8방향 이웃 평균 계산
        # -------------------------------------------------
        # 3x3 kernel에서 center는 제외
        neighbor_kernel = torch.ones(
            (C, 1, 3, 3),
            device=device,
            dtype=dtype
        )
        neighbor_kernel[:, :, 1, 1] = 0.0

        # zero padding 후 convolution
        # 경계 부분은 실제 존재하는 이웃 개수만큼만 평균내기 위해 count map을 따로 계산
        padded_feature = F.pad(feature, (1, 1, 1, 1), mode="constant", value=0.0)

        neighbor_sum = F.conv2d(
            padded_feature,
            neighbor_kernel,
            padding=0,
            groups=C
        )

        # 각 위치별 실제 이웃 개수 계산
        # corner: 3개, edge: 5개, inner: 8개
        valid_grid = torch.ones(
            (1, 1, H, W),
            device=device,
            dtype=dtype
        )

        count_kernel = torch.ones(
            (1, 1, 3, 3),
            device=device,
            dtype=dtype
        )
        count_kernel[:, :, 1, 1] = 0.0

        padded_valid = F.pad(valid_grid, (1, 1, 1, 1), mode="constant", value=0.0)

        neighbor_count = F.conv2d(
            padded_valid,
            count_kernel,
            padding=0
        )

        neighbor_mean = neighbor_sum / (neighbor_count + eps)

        # contrast map
        contrast = feature - neighbor_mean

        # -------------------------------------------------
        # 2. 같은 부호를 가진 인접 grid 개수 계산
        # -------------------------------------------------
        positive_mask = (contrast > 0).float()
        negative_mask = (contrast < 0).float()

        same_sign_kernel = torch.ones(
            (C, 1, 3, 3),
            device=device,
            dtype=dtype
        )
        same_sign_kernel[:, :, 1, 1] = 0.0

        pos_count = F.conv2d(
            F.pad(positive_mask, (1, 1, 1, 1), mode="constant", value=0.0),
            same_sign_kernel,
            padding=0,
            groups=C
        )

        neg_count = F.conv2d(
            F.pad(negative_mask, (1, 1, 1, 1), mode="constant", value=0.0),
            same_sign_kernel,
            padding=0,
            groups=C
        )

        same_sign_count = torch.where(
            contrast > 0,
            pos_count,
            torch.where(
                contrast < 0,
                neg_count,
                torch.zeros_like(contrast)
            )
        )

        # -------------------------------------------------
        # 3. edge map 계산
        # -------------------------------------------------
        # 규칙 그대로라면 signed_edge = contrast * same_sign_count
        signed_edge = contrast * same_sign_count

        # DropBlock 확률 p_map에 쓰려면 음수 edge는 부적절하므로 magnitude로 변환
        edge = signed_edge.abs()

        return edge

        
    def forward(self, x, edge_source):
        """
        x: dropout을 적용할 feature map [B, C, H, W]
        edge_source: edge map을 계산할 이전 feature map [B, C, H, W]
        """

        if (not self.training) and (not self.always_on):
            return x

        if edge_source is None:
            return x

        B, C, H, W = x.shape

        # 이전 feature map에서 edge map 계산
        edge = self.feature_to_edge_map(edge_source, eps=self.eps, detach=True)

        # 여기서 resize하지 않음
        # 대신 크기가 다르면 에러를 내서 구조를 명확히 확인
        if edge.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"edge map size {edge.shape[-2:]} and feature map size {x.shape[-2:]} do not match. "
                "현재 layer는 feature size가 바뀌는 layer일 수 있습니다."
            )

        score = self.eps + self.lambda_edge * edge
        score_mean = score.mean(dim=(2, 3), keepdim=True)

        p_map = self.gamma * score / (score_mean + self.eps)
        p_map = p_map.clamp(0.0, 1.0)

        center_mask = (torch.rand_like(p_map) < p_map).float()

        block_mask = F.max_pool2d(
            center_mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2
        )

        block_mask = block_mask.clamp(0.0, 1.0)

        keep_mask = 1.0 - block_mask

        keep_ratio = keep_mask.mean(dim=(1, 2, 3), keepdim=True)

        out = x * keep_mask / (keep_ratio + self.eps)

        return out

