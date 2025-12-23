import torch
import torch.nn as nn
import torch.nn.functional as F

class STE_Ceil(torch.autograd.Function):
    """Straight-Through Estimator for Ceil"""
    @staticmethod
    def forward(ctx, x_in):
        return torch.ceil(x_in)

    @staticmethod
    def backward(ctx, g):
        return g  # 梯度直通


class STE_Min(torch.autograd.Function):
    """Straight-Through Estimator for Min"""
    @staticmethod
    def forward(ctx, x_in1, x_in2, x_in3=None):
        def to_tensor(val, ref):
            return val if isinstance(val, torch.Tensor) else torch.tensor(val, dtype=ref.dtype, device=ref.device)

        t1 = to_tensor(x_in1, x_in2)
        t2 = to_tensor(x_in2, x_in2)
        t3 = to_tensor(x_in3 if x_in3 is not None else float('inf'), x_in2)

        return torch.min(torch.min(t1, t2), t3)

    @staticmethod
    def backward(ctx, g):
        return None, g, g


ste_ceil = STE_Ceil.apply
ste_min = STE_Min.apply

class CompressRate(nn.Module):
    """可微分的 Token 压缩率学习模块"""
    def __init__(self, patch_number=196, granularity=1):
        super().__init__()

        self.patch_number = patch_number
        self.class_token_num = 1
        self.temperature = 0.001

        # Token 保留数量候选：[196, 195, ..., 2, 1]
        candidates = torch.arange(patch_number, 0, -granularity).float()
        self.register_buffer('kept_token_candidate', candidates)

        # 可学习的保留率 logit
        initial_logit = self._ratio_to_logit(torch.tensor(0.9))
        self.ratio_logit = nn.Parameter(initial_logit, requires_grad=True)

        self.kept_token_number = patch_number + self.class_token_num
        self.update_kept_token_number()

    def _ratio_to_logit(self, ratio):
        """保留率 -> logit"""
        ratio = torch.clamp(ratio, 0.1, 0.9)
        return torch.log(ratio / (1.0 - ratio))

    def _logit_to_ratio(self, logit):
        """logit -> 保留率"""
        return torch.sigmoid(logit) * 0.8 + 0.1

    def _compute_distribution(self, ratio):
        """计算概率分布"""
        candidates_ratio = self.kept_token_candidate / self.patch_number
        distances = -(candidates_ratio - ratio).pow(2) / self.temperature
        return F.softmax(distances, dim=0)

    def update_kept_token_number(self):
        """更新保留的 Token 数量"""
        ratio = self._logit_to_ratio(self.ratio_logit)

        if self.training:   # 训练
            probs = self._compute_distribution(ratio)
            expected_kept = (self.kept_token_candidate * probs).sum()
            kept_num = ste_ceil(expected_kept) + self.class_token_num  # STE 使其可微

            self.kept_token_number = int(kept_num)
            return kept_num
        else:   # 推理
            target = ratio * self.patch_number
            idx = (self.kept_token_candidate - target).abs().argmin()
            kept_num = int(self.kept_token_candidate[idx]) + self.class_token_num

            self.kept_token_number = kept_num
            return kept_num

    def get_token_probability(self):
        """计算 Token 保留的概率"""
        ratio = self._logit_to_ratio(self.ratio_logit)
        probs = self._compute_distribution(ratio)

        total_tokens = self.patch_number + self.class_token_num
        token_prob = torch.zeros(total_tokens, device=self.ratio_logit.device)

        # 累积概率
        for kept_num, prob in zip(self.kept_token_candidate, probs):
            token_prob[:int(kept_num + self.class_token_num)] += prob

        return token_prob

    def get_token_mask(self, token_number=None):
        """生成 soft mask"""
        token_prob = self.get_token_probability()
        mask = torch.ones_like(token_prob)

        end_idx = int(token_number) if token_number else len(mask)
        mask[int(self.kept_token_number):end_idx] = 0

        return mask - token_prob.detach() + token_prob