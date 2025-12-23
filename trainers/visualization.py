"""训练可视化工具模块"""

import os
import numpy as np


class TrainingVisualizer:
    """训练过程可视化器"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.history = {
            'epochs': [],
            'loss_cls': [],
            'loss_flops': [],
            'flops_gflops': [],
            'kept_nums': {}
        }

    def record_batch(self, loss, loss_flops, flops_gflops):
        """记录单个batch的数据"""
        if not hasattr(self, 'epoch_losses_cls'):
            self.epoch_losses_cls = []
            self.epoch_losses_flops = []
            self.epoch_flops = []

        self.epoch_losses_cls.append(loss)
        self.epoch_losses_flops.append(loss_flops)
        self.epoch_flops.append(flops_gflops)

    def record_epoch(self, epoch, model):
        """记录epoch结束时的统计数据"""
        self.history['epochs'].append(epoch)
        self.history['loss_cls'].append(np.mean(self.epoch_losses_cls))
        self.history['loss_flops'].append(np.mean(self.epoch_losses_flops))
        self.history['flops_gflops'].append(np.mean(self.epoch_flops))

        # 记录kept_num
        for name, module in model.named_modules():
            if 'prune_ddp' in name and hasattr(module, 'kept_token_number'):
                layer_idx = int(name.split('.')[3])
                if layer_idx not in self.history['kept_nums']:
                    self.history['kept_nums'][layer_idx] = []
                self.history['kept_nums'][layer_idx].append(module.kept_token_number)

        # 重置batch级别的数据
        self.epoch_losses_cls = []
        self.epoch_losses_flops = []
        self.epoch_flops = []

    def generate_plots(self, target_flops=None):
        """生成训练可视化图表"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        if not self.history['epochs']:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        epochs = self.history['epochs']
        plot_style = {'linewidth': 2.5, 'markersize': 7}

        def setup_axis(ax, title, ylabel):
            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')

        # FLOPs演化
        axes[0, 0].plot(epochs, self.history['flops_gflops'], 'D-',
                        label='Actual FLOPs', color='#9467bd', **plot_style)
        if target_flops:
            axes[0, 0].axhline(y=target_flops, color='red', linestyle='--',
                               label=f'Target ({target_flops} GFLOPs)', linewidth=2)
        setup_axis(axes[0, 0], 'FLOPs Evolution', 'FLOPs (GFLOPs)')

        # 损失演化
        axes[0, 1].plot(epochs, self.history['loss_cls'], 'o-',
                        label='loss_cls', color='#1f77b4', **plot_style)
        axes[0, 1].plot(epochs, self.history['loss_flops'], 's-',
                        label='loss_flops', color='#ff7f0e', **plot_style)
        setup_axis(axes[0, 1], 'Loss Evolution', 'Loss')

        # Token数量和压缩率
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, (layer_idx, kept_nums) in enumerate(sorted(self.history['kept_nums'].items())):
            color = colors[i % len(colors)]
            axes[1, 0].plot(epochs, kept_nums, 'o-', label=f'Layer {layer_idx}',
                            color=color, **plot_style)
            axes[1, 1].plot(epochs, [(k-1)/196 for k in kept_nums], 'o-',
                            label=f'Layer {layer_idx}', color=color, **plot_style)

        setup_axis(axes[1, 0], 'Token Count per Layer', 'kept_num (Tokens)')
        setup_axis(axes[1, 1], 'Compression Ratio per Layer', 'Compression Ratio')
        axes[1, 1].set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_visualization.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
