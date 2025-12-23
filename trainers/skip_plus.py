import datetime
import math
import time
import torch
import torch.nn.functional as F
from scipy.stats import expon
from torch import nn
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms

from dassl.data.data_manager import build_data_loader, build_transform
from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import AverageMeter, MetricMeter

from torchinfo import summary

from .baselines.ft_clip import CustomCLIP, FinetuneCLIP
from .feature_loader import FeatureLoader
from .visualization import TrainingVisualizer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class Linear_Classifier(nn.Module):
    """ Linear Classifier 分类头"""
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout_rate=0.1, **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class FiLM(nn.Module):
    """特征调制"""
    def __init__(self,
                 dim,
                 bias=True,
                 use_sigmoid=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.has_bias = bias
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        scale = self.scale.unsqueeze(0).type(x.dtype)
        bias = self.bias.unsqueeze(0).type(x.dtype) if self.has_bias else None

        x = scale * x
        if bias is not None:
            x = x + bias

        if self.use_sigmoid:
            return x.sigmoid()

        return x


class PaddingTokenRemover:
    """移除文本 padding"""
    def __init__(self):
        self.eot_token = 49407  # end-of-text token

    def remove_padding_tokens(self, tokenized_texts):
        batch_size, seq_len = tokenized_texts.shape

        # 找到 EOT token 位置
        eot_mask = tokenized_texts == self.eot_token
        eot_positions = eot_mask.int().argmax(dim=1)
        has_eot = eot_mask.any(dim=1)
        lengths = torch.where(has_eot, eot_positions + 1, seq_len)

        # 截断到最大有效长度
        max_len = lengths.max().item()
        result = torch.zeros(batch_size, max_len, dtype=tokenized_texts.dtype, device=tokenized_texts.device)

        for i, length in enumerate(lengths):
            result[i, :length] = tokenized_texts[i, :length]

        return result

class TextTransformerWrapper(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.width = transformer.width
        self.layers = transformer.layers
        self.resblocks = nn.ModuleList(list(transformer.resblocks.children()))

    def forward(self, x, start_layer=None, end_layer=None, mask_len=None):
        start_layer = start_layer or 0
        end_layer = end_layer if end_layer is not None else len(self.resblocks) - 1

        if end_layer < start_layer:
            return x

        # 动态构建因果掩码
        if mask_len is not None:
            new_mask = self._build_causal_mask(mask_len, x.device, x.dtype)
            for block in self.resblocks:
                block.attn_mask = new_mask

        for block in self.resblocks[start_layer:end_layer + 1]:
            x = block(x)

        return x

    def _build_causal_mask(self, seq_len, device, dtype):
        """构建因果注意力掩码"""
        return torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype).triu_(1)

class TextEncoderWrapper(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.transformer = TextTransformerWrapper(text_encoder.transformer)
        self.token_embedding = text_encoder.token_embedding if hasattr(text_encoder, 'token_embedding') else None
        self.positional_embedding = text_encoder.positional_embedding
        self.ln_final = text_encoder.ln_final
        self.text_projection = text_encoder.text_projection
        self.dtype = text_encoder.dtype
        self.padding_remover = PaddingTokenRemover()

    def forward(self, x=None, start_layer=None, end_layer=None, tokenized_texts=None):
        if tokenized_texts is not None:
            compressed_texts = self.padding_remover.remove_padding_tokens(tokenized_texts)
        else:
            compressed_texts = None

        if start_layer is None:
            x, seq_length = self.pre_forward(compressed_texts)
        else:
            seq_length = x.shape[0]

        x = self.transformer(x, start_layer, end_layer, mask_len=seq_length)

        if end_layer is not None:
            return x

        return self.post_forward(x, compressed_texts)

    def pre_forward(self, compressed_texts):
        x = self.token_embedding(compressed_texts).type(self.dtype)
        seq_length = compressed_texts.shape[1]
        x = x + self.positional_embedding[:seq_length].type(self.dtype)
        return x.permute(1, 0, 2), seq_length

    def post_forward(self, x, texts):
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        eot_indices = texts.argmax(dim=-1)
        return x[torch.arange(x.shape[0]), eot_indices] @ self.text_projection


class ImageTransformerWrapper(nn.Module):
    def __init__(self, cfg, transformer):
        super().__init__()
        self.width = transformer.width
        self.layers = transformer.layers
        self.resblocks = nn.ModuleList(list(transformer.resblocks))
        self.start_layer = cfg.TRAINER.SKIP.START_LAYER
        self.end_layer = 11
        self._initialize_compress_rate_modules()

    def _initialize_compress_rate_modules(self):
        """初始化 CompressRate Token 压缩模块"""
        from clip.compress_rate import CompressRate
        for layer_idx in range(self.start_layer, self.end_layer + 1):
            self.resblocks[layer_idx].prune_ddp = CompressRate(patch_number=196, granularity=1)

    def forward(self, x, start_layer=None, end_layer=None, need_weights=False, do_token_compress=False):
        start_layer = start_layer or 0
        end_layer = end_layer if end_layer is not None else len(self.resblocks) - 1

        if end_layer < start_layer:
            return x

        last_attn_weights = None
        last_value_vectors = None
        policy = None  # Token 保留策略

        for i, block in enumerate(self.resblocks[start_layer:end_layer + 1]):
            layer_idx = start_layer + i
            # 在 start_layer 到 end_layer 之间进行 Token 压缩
            do_compress = do_token_compress and self.start_layer <= layer_idx <= self.end_layer
            x, attn_weights, policy, value_vectors = block(x, need_weights=need_weights, do_token_compress=do_compress, policy=policy)

            if layer_idx == end_layer:
                last_attn_weights = attn_weights
                last_value_vectors = value_vectors

        return x, last_attn_weights, last_value_vectors

class ImageEncoderWrapper(nn.Module):
    def __init__(self, cfg, image_encoder):
        super().__init__()
        self.input_resolution = image_encoder.input_resolution
        self.output_dim = image_encoder.output_dim
        self.conv1 = image_encoder.conv1
        self.class_embedding = image_encoder.class_embedding
        self.positional_embedding = image_encoder.positional_embedding
        self.ln_pre = image_encoder.ln_pre
        self.transformer = ImageTransformerWrapper(cfg, image_encoder.transformer)
        self.ln_post = image_encoder.ln_post
        self.proj = image_encoder.proj
        self.text_to_vision_mlp = nn.Linear(512, 768)
        self.class_prototypes = None

    def forward(self,
        x,
        start_layer=None,
        end_layer=None,
        do_token_compress=False,
        return_flops=False
    ):
        if start_layer is None:
            x = self.pre_forward(x)

        x, last_attn_weights, last_value_vectors = self.transformer(
            x,
            start_layer=start_layer,
            end_layer=end_layer,
            need_weights=do_token_compress,
            do_token_compress=do_token_compress
        )

        if end_layer is not None:
            return x

        x = x.permute(1, 0, 2)
        x = self.ln_post(x).type(self.proj.dtype).to(self.proj.device)
        cls_feature = x[:, 0, :] @ self.proj  # CLS token 特征

        # 提取局部 Token 特征（用于 CAT 头）
        local_features = None
        if do_token_compress:
            local_features = x[:, 1:, :] @ self.proj
            token_scores = self.compute_token_scores(last_attn_weights, last_value_vectors)
            local_features = self.select_topk_tokens(local_features, token_scores)

        flops = self.calculate_flops(start_layer, end_layer) if return_flops else None
        return cls_feature, local_features, flops

    def pre_forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls_token = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        x = self.ln_pre(x + self.positional_embedding.type(x.dtype))
        return x.permute(1, 0, 2)

    def compute_token_scores(self, attn_weights, value_vectors=None):
        cls_attn = attn_weights[:, 0, 1:]  # [B, S-1]

        if value_vectors is not None:
            value_vectors_transposed = value_vectors.transpose(0, 1)  # [B, S, D]
            value_norms = torch.norm(value_vectors_transposed[:, 1:, :], dim=2)  # [B, S-1]
            weighted_scores = cls_attn * value_norms  # 注意力权重 × Value 模长
            token_scores = F.normalize(weighted_scores, p=1, dim=1)
        else:
            token_scores = F.normalize(cls_attn, p=1, dim=1)

        return token_scores

    def select_topk_tokens(self, local_features, token_scores, keep_num=10):
        """选择 Top-K 重要的 Token"""
        batch_size, num_tokens, dim = local_features.shape
        patch_scores = token_scores[:, 1:num_tokens+1]
        k = min(keep_num, num_tokens)
        topk_idx = torch.topk(patch_scores, k, dim=1).indices
        batch_indices = torch.arange(batch_size, device=local_features.device).unsqueeze(1)
        return local_features[batch_indices, topk_idx]

    def calculate_flops(self, start_layer, end_layer):
        start_layer = start_layer or 0
        end_layer = end_layer if end_layer is not None else len(self.transformer.resblocks) - 1
        C = self.transformer.width

        if self.training:
            from clip.compress_rate import ste_min

            device = self.transformer.resblocks[self.transformer.start_layer].prune_ddp.ratio_logit.device

            N = torch.tensor(197.0, dtype=torch.float32, device=device)
            flops = torch.tensor(0.0, dtype=torch.float32, device=device)

            with torch.amp.autocast(device_type='cuda', enabled=False):
                # 0 -- start_layer
                for layer_idx in range(0, start_layer):
                    flops += 4*N*C*C + 2*N*N*C + 8*N*C*C

                # start_layer -- end_layer
                for layer_idx in range(start_layer, end_layer + 1):
                    flops += 4*N*C*C + 2*N*N*C  # Attention

                    block = self.transformer.resblocks[layer_idx]
                    kept_num = block.prune_ddp.update_kept_token_number()
                    N = ste_min(N, kept_num.float(), torch.tensor(float('inf'), device=device))  # 使用 STE 更新 Token 数

                    flops += 8*N*C*C  # MLP
        else:
            N = 197.0
            flops = 0.0

            for layer_idx in range(0, start_layer):
                flops += 4*N*C*C + 2*N*N*C + 8*N*C*C

            for layer_idx in range(start_layer, end_layer + 1):
                flops += 4*N*C*C + 2*N*N*C

                block = self.transformer.resblocks[layer_idx]
                N = float(block.prune_ddp.kept_token_number)

                flops += 8*N*C*C

        return flops

class SkipPlusCustomCLIP(CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        self.start_layer = cfg.TRAINER.SKIP.START_LAYER
        
        super().__init__(cfg, classnames, clip_model)
        self.num_classes = len(classnames)
        self.top_ratio = cfg.TRAINER.SKIP.TOP_RATIO
        self.max_top = cfg.TRAINER.SKIP.MAX_TOP

        self.text_encoder = TextEncoderWrapper(self.text_encoder)
        self.image_encoder = ImageEncoderWrapper(cfg, self.image_encoder)

        self.add_learnable_parameter('text_encoder', self.text_encoder)
        self.add_learnable_parameter('image_encoder', self.image_encoder)
        self.set_exclude_parameters()
        
        self.text_features = None
        self.class_prototypes = None
        self.reserve_mask = None
        self.tokenized_texts = self.prompt_learner()
        self.dataset_name = self.cfg.DATASET.NAME
        self.subsample_classes = cfg.DATASET.SUBSAMPLE_CLASSES
        clip_dim = clip_model.text_projection.size(1)
        
        # FiLM 特征调制层
        self.film_img = FiLM(clip_dim)
        self.film_text = FiLM(clip_dim)

        # CAT 分类头
        if (self.subsample_classes == 'base') \
        or (self.subsample_classes == 'all' and 'ImageNet' in self.dataset):
            self.cat_head = Linear_Classifier(input_dim=clip_dim, output_dim=len(classnames)).type(self.dtype)
        else:
            self.cat_head = nn.Identity()

        self.add_learnable_parameter('film_img', self.film_img)
        self.add_learnable_parameter('film_text', self.film_text)
        self.add_learnable_parameter('cat_head', self.cat_head)

        # 双头权重：ITM + CAT
        self.itm_weight = cfg.TRAINER.SKIP.ITM_WEIGHT
        self.cat_weight = 1.0 - self.itm_weight

    def forward(self, images, labels=None, return_flops=False):
        do_token_compress = (self.subsample_classes == 'base') or \
                       (self.subsample_classes == 'all' and 'ImageNet' in self.dataset_name)

        images = images.type(self.dtype)

        if labels is not None:  # 训练模式
            cls_feature, local_token_features, flops = self.image_encoder(
                images,
                start_layer=self.start_layer,
                do_token_compress=do_token_compress,
                return_flops=return_flops
            )

            # 动态类别选择
            tokenized_texts, text_features, labels, selected_class_indices = self.select_classes(
                cls_feature, self.tokenized_texts, self.text_features, labels
            )

            text_features = self.text_encoder(text_features, start_layer=self.start_layer, tokenized_texts=tokenized_texts)
        else:  # 推理模式
            text_features = self.text_encoder(tokenized_texts=self.tokenized_texts)
            cls_feature, local_token_features, _ = self.image_encoder(
                images,
                do_token_compress=do_token_compress
            )
            selected_class_indices = None
            flops = None

        # 双头分类
        if do_token_compress:
            result = self.forward_base(cls_feature, local_token_features, text_features, labels, selected_class_indices)
        else:
            result = self.forward_new(cls_feature, text_features, labels)

        if labels is not None:
            loss, loss_summary = result
            return loss, loss_summary, flops
        else:
            return result
    
    def forward_base(self, cls_feature, local_token_features, text_features, labels=None, selected_class_indices=None):
        """Base 类别的双头分类（ITM + CAT）"""
        itm_logits = self.forward_itm_head(cls_feature, text_features)
        cat_logits, cat_labels = self.forward_cat_head(local_token_features, text_features, labels)

        cat_scale_factor = 9.0
        cat_logits = cat_logits * cat_scale_factor

        if labels is not None:
            cat_labels = selected_class_indices[cat_labels]
            return self.compute_loss(itm_logits, cat_logits, labels, cat_labels)
        else:
            final_logits = self.itm_weight * itm_logits + self.cat_weight * cat_logits
            return final_logits            
    
    def forward_new(self, cls_feature, text_features, labels=None):
        itm_logits = self.forward_itm_head(cls_feature, text_features)
        return itm_logits
    
    def forward_itm_head(self, cls_feature, text_features):
        cls_feature_norm = cls_feature / cls_feature.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        itm_logits = self.logit_scale.exp() * cls_feature_norm @ text_features_norm.t()  # [B, C]
        return itm_logits
        
    def forward_cat_head(self, local_token_features, text_features, labels, selected_class_indices=None):
        """CAT 分类头：使用局部 Token 特征进行分类"""
        text_feats = self.film_text(text_features)  # [C, D]
        img_feats = self.film_img(local_token_features)  # [B, S, D]
        batch_size, num_tokens, feature_dim = img_feats.shape

        if labels is None:  # 推理模式
            all_feats = img_feats.reshape(batch_size * num_tokens, feature_dim)
            all_logits = self.cat_head(all_feats).reshape(batch_size, num_tokens, -1)   #[B, S, num_classes]

            # 加权聚合
            probs = torch.softmax(all_logits, dim=-1)
            weights = torch.softmax(probs.max(dim=-1)[0], dim=1).unsqueeze(-1)
            all_logits = (all_logits * weights).sum(dim=1)

            all_labels = None

        else:  # 训练模式
            text_feats = text_feats[labels]  # [B, D]
            img_feats_flat = img_feats.reshape(batch_size * num_tokens, feature_dim)

            img_logits = self.cat_head(img_feats_flat)  # [B*S, num_classes]
            text_logits = self.cat_head(text_feats)  # [B, num_classes]

            # 拼接文本特征和图像特征
            all_logits = torch.cat([text_logits, img_logits])  # [B + B*S, num_classes]
            all_labels = torch.cat([labels, labels.unsqueeze(1).repeat(1, num_tokens).reshape(-1)])  # [B + B*S]

        return all_logits, all_labels


    def compute_loss(self, itm_logits, cat_logits, labels, cat_labels):
        loss_itm = F.cross_entropy(itm_logits, labels)
        loss_cat = F.cross_entropy(cat_logits, cat_labels)
        total_loss = self.itm_weight * loss_itm + self.cat_weight * loss_cat

        loss_summary = {
            'loss': total_loss,
            'acc': compute_accuracy(itm_logits, labels)[0]
        }

        return total_loss, loss_summary

    @torch.no_grad()
    def select_classes(self, image_features, tokenized_texts, text_features, labels):
        num_tops = math.ceil(self.top_ratio * self.num_classes) \
                   if self.top_ratio <= 1.0 else math.ceil(min(self.top_ratio, self.num_classes))
        num_tops = min(num_tops, self.max_top)
        
        if self.top_ratio == 1.0 or num_tops == self.num_classes:
            all_indices = torch.arange(self.num_classes).to(labels.device)
            return tokenized_texts, text_features, labels, all_indices
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        class_prototypes = self.class_prototypes
        # (B, D) @ (D, C) -> (B, C)
        similarity = image_features @ class_prototypes.t()
        similarity[torch.arange(similarity.shape[0]), labels] = 1e4
        max_similarity, _ = similarity.max(dim=0)
        _, inds = max_similarity.sort(descending=True)

        if self.reserve_mask is None:
            x = torch.linspace(0.0, 5.0, steps=self.num_classes - num_tops)
            
            assert self.cfg.TRAINER.SKIP.LAMBDA > 0
            pdf = expon.pdf(x, scale=self.cfg.TRAINER.SKIP.LAMBDA)
            pdf = (pdf - pdf.min()) / (pdf.max() - pdf.min())
            reserve_ratios = [1.0] * num_tops + pdf.tolist()
            reserve_ratios = torch.tensor(reserve_ratios).to(max_similarity.device)
            self.reserve_mask = torch.rand_like(max_similarity) < reserve_ratios

        inds_filtered = inds[self.reserve_mask]
        inds, _ = inds_filtered.sort()

        # select text features
        # (C, L) -> (K', L)
        tokenized_texts = tokenized_texts[inds]
        # (L, C, D) -> (L, K', D)
        text_features = text_features[:, inds]

        # select labels
        labels_one_hot = F.one_hot(labels, self.num_classes)
        labels_filtered_one_hot = labels_one_hot[:, inds]
        new_labels = labels_filtered_one_hot.argmax(dim=1)
        
        return tokenized_texts, text_features, new_labels, inds
    
    def forward_image_features(self, images, labels=None):
        images = self.image_encoder.pre_forward(images.type(self.dtype))
        image_features_medium = self.image_encoder(images, start_layer=0, end_layer=self.start_layer - 1)
        return image_features_medium

        
    def forward_text_features(self):
        compressed_texts = self.text_encoder.padding_remover.remove_padding_tokens(self.tokenized_texts)
        x, seq_length = self.text_encoder.pre_forward(compressed_texts)
        text_features_medium = self.text_encoder.transformer(x, start_layer=0, end_layer=self.start_layer - 1, mask_len=seq_length)
        text_features_out = self.text_encoder(tokenized_texts=self.tokenized_texts)
        
        return text_features_medium, text_features_out


    def drop_layers_before_start(self):
        for layer in range(self.start_layer):
            self.text_encoder.transformer.resblocks[layer] = nn.Identity()
            self.image_encoder.transformer.resblocks[layer] = nn.Identity()
        
        del self.text_encoder.positional_embedding
        del self.text_encoder.token_embedding
        del self.image_encoder.positional_embedding
        del self.image_encoder.class_embedding
        del self.image_encoder.conv1
    
    def set_exclude_parameters(self):
        self.add_excluded_parameter('token_embedding', self.text_encoder.token_embedding)
        self.add_excluded_parameter('text_positional_embedding', self.text_encoder.positional_embedding)
        self.add_excluded_parameter('class_embedding', self.image_encoder.class_embedding)
        self.add_excluded_parameter('conv1', self.image_encoder.conv1)
        self.add_excluded_parameter('ln_pre', self.image_encoder.ln_pre)
        self.add_excluded_parameter('image_positional_embedding', self.image_encoder.positional_embedding)

        for layer in range(self.start_layer):
            self.add_excluded_parameter(f'text_encoder_resblocks{layer}', self.text_encoder.transformer.resblocks[layer])
            self.add_excluded_parameter(f'image_encoder_resblocks{layer}', self.image_encoder.transformer.resblocks[layer])



@TRAINER_REGISTRY.register()
class SkipPlus(FinetuneCLIP):
    def __init__(self, cfg):
        if cfg.DATASET.NAME in ['ImageNet', 'SUN397', 'Food101']:
            cfg.defrost()
            cfg.OPTIM.MAX_EPOCH = int(cfg.OPTIM.MAX_EPOCH / 4)
            cfg.freeze()

        super().__init__(cfg)

    def build_custom_clip(self, cfg, classnames, clip_model):
        self.model = SkipPlusCustomCLIP(cfg, classnames, clip_model)
        self.num_classes = len(classnames)

    @torch.no_grad()
    def before_train(self):
        self.model.tokenized_texts = self.model.tokenized_texts.to(self.device)
        
        super().before_train()

        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        mode = 'batch_random' if self.cfg.DATASET.NUM_SHOTS >= 4 else 'random'
        self.feature_loader = FeatureLoader(batch_size, mode)

        with torch.no_grad():
            print('Extracting text features...')
            text_features_medium, text_features_out = self.extract_text_features()

            print('Extracting image features...')
            image_features_medium, labels = self.extract_image_features(0)

            self.feature_loader.set_features(image_features_medium, labels)
        
        self.model.text_features = text_features_medium.to(self.device)
        class_prototypes = text_features_out / text_features_out.norm(dim=-1, keepdim=True)
        self.model.class_prototypes = class_prototypes.to(self.device)
        self.model.image_encoder.class_prototypes = class_prototypes.to(self.device)

        torch.cuda.empty_cache()
        self.time_start = time.time()

        if self.cfg.TRAINER.SKIP.USE_FLOPS_LOSS:
            self.visualizer = TrainingVisualizer(self.cfg.OUTPUT_DIR)
    
    @torch.no_grad()
    def before_test(self):
        for name, module in self.model.named_modules():
            if hasattr(module, 'prune_ddp') and module.prune_ddp is not None:
                module.prune_ddp.update_kept_token_number()

        super().before_test()
    
    @torch.no_grad()
    def before_epoch(self):
        super().before_epoch()

        if self.epoch != 0:
            text_features_medium, _ = self.extract_text_features()
            image_features_medium, labels = self.extract_image_features(self.epoch)
            self.model.text_features = text_features_medium.to(self.device)
            self.feature_loader.set_features(image_features_medium, labels)
    
    @torch.no_grad()
    def after_epoch(self):
        super().after_epoch()

        if self.cfg.TRAINER.SKIP.USE_FLOPS_LOSS:
            self.visualizer.record_epoch(self.epoch, self.model)


    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.feature_loader)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.feature_loader.get_features()):
            data_time.update(time.time() - end)

            self.model.batch_idx = self.batch_idx

            loss_summary = self.forward_backward(batch)

            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if self.cfg.TRAIN.PRINT_FREQ > 0:
                meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
                only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += self.num_batches - self.batch_idx - 1
                    nb_remain += (
                        self.max_epoch - self.epoch - 1
                    ) * self.num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                    info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"{losses}"]
                    info += [f"lr {self.get_current_lr():.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()


    def forward_backward(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        use_flops_loss = self.cfg.TRAINER.SKIP.USE_FLOPS_LOSS
        loss, loss_summary, flops = self.model(images, labels, return_flops=use_flops_loss)

        if use_flops_loss:
            flops_gflops = flops / 1e9
            loss_flops = (flops_gflops - self.cfg.TRAINER.SKIP.TARGET_FLOPS) ** 2

            warmup_epochs = self.cfg.TRAINER.SKIP.FLOPS_WARMUP_EPOCHS
            lamb = 0.0 if self.epoch < warmup_epochs else self.cfg.TRAINER.SKIP.FLOPS_LOSS_WEIGHT

            total_loss = loss + lamb * loss_flops

            self.visualizer.record_batch(loss.item(), loss_flops.item(), flops_gflops.item())

            self.model_backward_and_update(total_loss)
        else:
            self.model_backward_and_update(loss)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    @torch.no_grad()
    def after_train(self):
        super().after_train()

        if self.cfg.TRAINER.SKIP.USE_FLOPS_LOSS:
            target_flops = getattr(self.cfg.TRAINER.SKIP, 'TARGET_FLOPS', None)
            self.visualizer.generate_plots(target_flops)
    
    @torch.no_grad()
    def extract_text_features(self):
        text_features_medium, text_features_out = self.model.forward_text_features()
        return text_features_medium.cpu(), text_features_out.cpu()
    
    @torch.no_grad()
    def extract_image_features(self, epoch):
        tfm_train = build_transform(self.cfg, is_train=True)
        quick_loader = build_data_loader(
            self.cfg, sampler_type='SequentialSampler',
            data_source=self.dm.dataset.train_x,
            batch_size=self.cfg.TRAINER.SKIP.QUICK_BATCH_SIZE,
            n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train, is_train=False, dataset_wrapper=None
        ) 

        image_features_medium_list = []
        labels_list = []

        for batch in tqdm(quick_loader, desc=f'Extracting image features for epoch {epoch}'):
            images, labels_ = self.parse_batch_train(batch)
            image_features_medium_ = self.model.forward_image_features(images, labels=labels_)

            image_features_medium_list.append(image_features_medium_.cpu())
            labels_list.append(labels_.cpu())

        image_features_medium = torch.cat(image_features_medium_list, dim=1)
        labels = torch.cat(labels_list, dim=0)

        return image_features_medium, labels
    

    def model_inference(self, input):
        return self.model(input)
    
    def model_summary(self):
        vision_dim, text_dim = 768, 512
        vision_len = 1 + 14 * 14
        text_len = self.model.text_encoder.padding_remover.remove_padding_tokens(self.model.tokenized_texts).shape[1]
            
        self.model.text_features = torch.randn(text_len, self.num_classes, text_dim).to(self.device).type(self.model.dtype)
        self.model.class_prototypes = torch.randn(self.num_classes, text_dim).to(self.device).type(self.model.dtype)
        
        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE

        dummy_images = torch.randn(vision_len, batch_size, vision_dim).to(self.device).type(self.model.dtype)
        dummy_labels = torch.randint(low=0, high=self.num_classes, size=(batch_size,), device=self.device)
        summary(self.model, input_data=(dummy_images, dummy_labels), device=self.device, mode='train')


