# Skip-Plus

---

## 安装指南

### 安装步骤

**1. 创建 Conda 环境**

```bash
conda create -n skip-plus python=3.10
conda activate skip-plus
```

**2. 安装 PyTorch**

```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**3. 安装 Dassl 库**

```bash
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/
pip install -r requirements.txt
python setup.py develop
cd ..
```

**4. 安装 Skip-Plus**

```bash
git clone https://github.com/JXuanYu/Skip-Plus.git
cd Skip-Plus/
pip install -r requirements.txt
```

**5. 下载数据集**

请参考 [datasets/DATASETS.md](datasets/DATASETS.md) 准备数据集。

---

## 训练与评估

本项目提供并行运行脚本 `parallel_runner.py`，支持多种 Prompt 方法（CoOp、CoCoOp、ProGrad、KgCoOp、MaPLe、PromptSRC、TCP、KgDePT、CoPrompt）和 Adapter 方法（CLIP-Adapter）。

### 1. 配置路径

编辑 `configs.py` 设置数据集和输出路径：

```python
base = dict(
    # 数据集配置
    data = dict(
        root=f'{ROOT}/datasets/DATA',
        datasets_base_to_new=['imagenet', 'stanford_cars', ...],
    ),

    # 邮件通知配置（可选）
    mail = dict(
        username='your@mail.com',
        password='your_password',
        host='smtp.example.com',
        to='your@mail.com',
    ),

    # 输出配置
    output = dict(
        root=f'{ROOT}/outputs',
        result=f'{ROOT}/results/acc',
        cost=f'{ROOT}/results/cost',
        remove_dirs=[],
    ),
)
```

### 2. 配置任务流水线

在 `configs.py` 中配置并行任务：

```python
pipeline = [
    # 流水线将并行运行
    # Pipeline 1
    dict(
        gpu_ids=[0, 1, 2],  # 该流水线使用的 GPU
        tasks=[
            'coop',
            'ftclip',
            'skip_plus',
        ]
    ),
    # Pipeline 2
    dict(
        gpu_ids=[3, 4, 5],
        tasks=[
            'skip_plus',
        ]
    )
]
```

### 3. 运行训练

```bash
python parallel_runner.py
```

运行完成后：
- 模型输出保存在 `outputs/`
- 准确率结果保存在 `results/acc/`
- 计算成本结果保存在 `results/cost/`
- 训练摘要将发送至配置的邮箱（如已配置）

---

## 致谢

本项目基于以下优秀工作：
- [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)
- [CLIP](https://github.com/openai/CLIP)
- [SkipTuning](https://github.com/Koorye/SkipTuning)
