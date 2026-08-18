import copy
import os
from deepmerge import Merger

ROOT = "/your/project/root/here"

base = dict(
    data = dict(
        root=f'{ROOT}/datasets/DATA',
        datasets_base_to_new=['imagenet', 'sun397', 'stanford_cars', 'oxford_flowers', 'food101', 'ucf101', 
                              'caltech101', 'fgvc_aircraft', 'dtd', 'oxford_pets', 'eurosat'],
        datasets_cross_dataset=['caltech101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 'food101',
                                'fgvc_aircraft', 'sun397', 'dtd', 'eurosat', 'ucf101',
                                'imagenetv2', 'imagenet_sketch', 'imagenet_a', 'imagenet_r'],
        datasets_all=['imagenet', 'sun397', 'stanford_cars', 'oxford_flowers', 'food101', 'ucf101', 
                      'caltech101', 'fgvc_aircraft', 'dtd', 'oxford_pets', 'eurosat'],
    ),

    mail = dict(
        username='disabled',
        password='disabled',
        host='disabled',
        to='disabled'
    ),

    # training configs
    train = dict(
        mode='b2n',
        seeds=[1,2,3],
        load_from='',
        loadep=-1,
        shots=16,
        opts=[],
    ),

    # grid search configs
    grid_search = dict(
        plot='line',
        mode='sequential',
        params=[]
    ),

    # output configs
    output = dict(
        root=f'{ROOT}/outputs',
        result=f'{ROOT}/results/acc',
        cost=f'{ROOT}/results/cost',
        remove_dirs=[],
    ),
)

#####################################################
# Base-to-New Generalization
skip_plus = dict(
    train = dict(
        trainer='SkipPlus',
        cfg='vit_b16_bs4',
    ),
)

# Cross Dataset Transfer & Domain Generalization
skip_plus_xd = dict(
    train = dict(
        mode='xd',
        trainer='SkipPlus',
        cfg='vit_b16_bs4_cross_datasets',
    ),
)

# Few-shot Learning
skip_plus_all = dict(
    train = dict(
        mode='all',
        trainer='SkipPlus',
        cfg='vit_b16_bs4_few_shot',
    ),

    grid_search = dict(
        params=[
            dict(
                name='DATASET.NUM_SHOTS',
                alias='shot',
                values=[16, 8, 4, 2, 1],
            ),
        ],
    )
)



#####################################################
# Ablation study

skip_plus_layer = dict(
    train = dict(
        trainer='SkipPlus',
        cfg='vit_b16_bs4',
    ),
    
    grid_search = dict(
        params=[
            dict(
                name='TRAINER.SKIP.START_LAYER',
                alias='layer',
                values=[2, 4, 6, 8, 10],
            )
        ]
    )
)

skip_plus_top = dict(
    train = dict(
        trainer='SkipPlus',
        cfg='vit_b16_bs4',
    ),
    
    grid_search = dict(
        params=[
            dict(
                name='TRAINER.SKIP.TOP_RATIO',
                alias='top',
                values=[1.0, 0.9, 0.7, 0.5, 0.3, 0.1],
            )
        ]
    )
)

skip_plus_itm_weight_ablation = dict(
    train = dict(
        trainer='SkipPlus',
        cfg='vit_b16_bs4',
    ),

    grid_search = dict(
        params=[
            dict(
                name='TRAINER.SKIP.ITM_WEIGHT',
                alias='itm_w',
                values=[0.1, 0.3, 0.5, 0.65, 0.7, 0.9]
            )
        ]
    ),
)

skip_plus_flops_loss_weight_ablation = dict(
    train = dict(
        trainer='SkipPlus',
        cfg='vit_b16_bs4',
    ),

    grid_search = dict(
        params=[
            dict(
                name='TRAINER.SKIP.FLOPS_LOSS_WEIGHT',
                alias='flops_loss_w',
                values=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9], 
            )
        ]
    ),
)

skip_plus_tskip_dual_head_component_ablation = dict(
    train = dict(
        trainer='SkipPlus',
        cfg='vit_b16_bs4',
    ),

    grid_search = dict(
        mode='sequential',
        params=[
            dict(
                name='TRAINER.SKIP.USE_TSKIP',
                alias='tskip',
                values=[True, False, True, False],
            ),
            dict(
                name='TRAINER.SKIP.USE_DUAL_HEAD',
                alias='dual',
                values=[True, True, False, False],
            ),
        ]
    ),
)

#####################################################

pipeline = [
    dict(
        gpu_ids=[0,1],
        tasks=[
            'skip_plus',
            'skip_plus_xd',
            'skip_plus_all',
            'skip_plus_layer',
            'skip_plus_top',
            'skip_plus_itm_weight_ablation',
            'skip_plus_flops_loss_weight_ablation',
            'skip_plus_tskip_dual_head_component_ablation',
        ]   
    )
]

#####################################################


def get_pipeline():
    global base, pipeline

    pipeline = copy.deepcopy(pipeline)
    merger = Merger([(list, ['override']), (dict, ['merge']), (set, ['override'])],
                    ['override'], ['override'])

    for pipe in pipeline:
        tasks = []

        for task in pipe['tasks']:
            base_cfg = copy.deepcopy(base)
            if isinstance(task, str):
                cfg = copy.deepcopy(eval(task))
                task_name = task
            elif isinstance(task, dict):
                cfg = copy.deepcopy(task)
                task_name = cfg.get('name', cfg.get('train', {}).get('trainer', 'custom_task'))
            else:
                raise TypeError(f"Unsupported task type: {type(task)}")

            cfg = merger.merge(base_cfg, cfg)
            cfg['gpu_ids'] = pipe['gpu_ids']
            cfg['name'] = task_name
            tasks.append(copy.deepcopy(cfg))

        pipe['tasks'] = tasks

    return pipeline
