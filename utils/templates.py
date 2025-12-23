# 0: data root
# 1: seed
# 2: trainer
# 3: dataset
# 4: cfg
# 5: root
# 6: shots
# 7: load epoch
TRAIN_CMD_TEMPLATE_BASE_TO_NEW = r'''python train.py \
--root {0} \
--seed {1} \
--trainer {2} \
--dataset-config-file configs/datasets/{3}.yaml \
--config-file configs/trainers/{2}/{4}.yaml \
--output-dir {5}/train_base/{2}/{3}/shots{6}/{4}/seed{1} \
DATASET.NUM_SHOTS {6} DATASET.SUBSAMPLE_CLASSES base '''

TEST_CMD_TEMPLATE_BASE_TO_NEW = r'''python train.py \
--root {0} \
--seed {1} \
--trainer {2} \
--dataset-config-file configs/datasets/{3}.yaml \
--config-file configs/trainers/{2}/{4}.yaml \
--output-dir {5}/test_new/{2}/{3}/shots{6}/{4}/seed{1} \
--model-dir {5}/train_base/{2}/{3}/shots{6}/{4}/seed{1} \
--load-epoch {7} \
--eval-only \
DATASET.NUM_SHOTS {6} DATASET.SUBSAMPLE_CLASSES new '''

TEST_CMD_TEMPLATE_BASE_ONLY = r'''python train.py \
--root {0} \
--seed {1} \
--trainer {2} \
--dataset-config-file configs/datasets/{3}.yaml \
--config-file configs/trainers/{2}/{4}.yaml \
--output-dir {5}/test_base/{2}/{3}/shots{6}/{4}/seed{1} \
--model-dir {5}/train_base/{2}/{3}/shots{6}/{4}/seed{1} \
--load-epoch {7} \
--eval-only \
DATASET.NUM_SHOTS {6} DATASET.SUBSAMPLE_CLASSES base '''

TRAIN_CMD_TEMPLATE_ALL = r'''python train.py \
--root {0} \
--seed {1} \
--trainer {2} \
--dataset-config-file configs/datasets/{3}.yaml \
--config-file configs/trainers/{2}/{4}.yaml \
--output-dir {5}/train_base/{2}/{3}/shots{6}/{4}/seed{1} \
DATASET.NUM_SHOTS {6} DATASET.SUBSAMPLE_CLASSES all '''

# 0: data root
# 1: seed
# 2: trainer
# 3: dataset
# 4: cfg
# 5: root
# 6: shots
# 7: load dataset
# 8: load epoch
TRAIN_CMD_TEMPLATE_CROSS_DATASET = r'''python train.py \
--root {0} \
--seed {1} \
--trainer {2} \
--dataset-config-file configs/datasets/{3}.yaml \
--config-file configs/trainers/{2}/{4}.yaml \
--output-dir {5}/{2}/{3}/shots{6}/{4}/seed{1} \
DATASET.NUM_SHOTS {6} DATASET.SUBSAMPLE_CLASSES all '''

TEST_CMD_TEMPLATE_CROSS_DATASET = r'''python train.py \
--root {0} \
--seed {1} \
--trainer {2} \
--dataset-config-file configs/datasets/{3}.yaml \
--config-file configs/trainers/{2}/{4}.yaml \
--output-dir {5}/{2}/{3}/shots{6}/{4}/seed{1} \
--model-dir {5}/{2}/{7}/shots{6}/{4}/seed{1} \
--load-epoch {8} \
--eval-only \
DATASET.NUM_SHOTS {6} DATASET.SUBSAMPLE_CLASSES all '''


def get_command(data_root, seed, trainer, dataset, cfg, root, shots, load_dataset, load_epoch, opts=[], mode='b2n', train=True):
    if mode == 'all':
        cmd = TRAIN_CMD_TEMPLATE_ALL.format(data_root, seed, trainer, dataset, cfg, root, shots)
    elif mode == 'b2n':
        if train:
            cmd = TRAIN_CMD_TEMPLATE_BASE_TO_NEW.format(data_root, seed, trainer, dataset, cfg, root, shots)
        else:
            cmd = TEST_CMD_TEMPLATE_BASE_TO_NEW.format(data_root, seed, trainer, dataset, cfg, root, shots, load_epoch)
    elif mode == 'base_only':
        # 新增：只测试base类别的模式
        if train:
            cmd = TRAIN_CMD_TEMPLATE_BASE_TO_NEW.format(data_root, seed, trainer, dataset, cfg, root, shots)
        else:
            cmd = TEST_CMD_TEMPLATE_BASE_ONLY.format(data_root, seed, trainer, dataset, cfg, root, shots, load_epoch)
    elif mode == 'xd':
        if train:
            cmd = TRAIN_CMD_TEMPLATE_CROSS_DATASET.format(data_root, seed, trainer, dataset, cfg, root, shots)
        else:
            cmd = TEST_CMD_TEMPLATE_CROSS_DATASET.format(data_root, seed, trainer, dataset, cfg, root, shots, load_dataset, load_epoch)
    else:
        raise ValueError(f"Unknown mode: {mode}")
            
    for opt in opts:
        # 确保参数值被正确引用，特别是包含特殊字符的值
        if isinstance(opt, str) and ('(' in opt or ')' in opt or ' ' in opt):
            cmd += f'"{opt}" '
        else:
            cmd += f'{opt} '
        
    return cmd
