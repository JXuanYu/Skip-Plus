import copy
import os
from deepmerge import Merger

# Get project root directory
ROOT = os.path.dirname(os.path.abspath(__file__))

base = dict(
    data = dict(
        root=f'{ROOT}/datasets/DATA',
        datasets_base_to_new=['imagenet', 'stanford_cars', 'oxford_flowers',  'ucf101','caltech101', 'fgvc_aircraft', 'dtd', 'oxford_pets', 'eurosat','food101','sun397'],
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

clip_adapter_origin = dict(
    train = dict(
        trainer='OriginCLIPAdapter',
        cfg='vit_b16_ep100',
    ),
)

clip_adapter = dict(
    train = dict(
        trainer='CLIPAdapter',
        cfg='vit_b16_ep100',
    ),
)

clip_adapter_bs4 = dict(
    train = dict(
        trainer='CLIPAdapter',
        cfg='vit_b16_ep10_bs4',
    ),
)

coop = dict(
    train = dict(
        trainer='CoOp',
        cfg='vit_b16_ep100',
    ),
)

cocoop = dict(
    train = dict(
        trainer='CoCoOp',
        cfg='vit_b16_c4_ep10_batch1_ctxv1',
    ),
)

prograd = dict(
    train = dict(
        trainer='ProGrad',
        cfg='vit_b16_ep100_ctx',
    ),
)

kgcoop = dict(
    train = dict(
        trainer='KgCoOp',
        cfg='vit_b16_ep100_ctxv1',
    ),
)

maple = dict(
    train = dict(
        trainer='MaPLe',
        cfg='vit_b16_c2_ep5_batch4_2ctx',
    ),
)

tcp = dict(
    train = dict(
        trainer='TCP',
        cfg='vit_b16_ep100_ctxv1',
    ),
)

promptsrc = dict(
    train = dict(
        trainer='PromptSRC',
        cfg='vit_b16_c2_ep20_batch4_4+4ctx',
    ),
)

kgdept = dict(
    train = dict(
        trainer='KgDePT',              
        cfg='vit_b16_ep10_ctxv1_bs4_lr35', 
    ),
)

coprompt = dict(
    train = dict(
        trainer='CoPrompt',
        cfg='coprompt',
    ),
)

ftclip = dict(
    train = dict(
        trainer='FinetuneCLIP',
        cfg='vit_b16_ep10_bs4',
    ),
)

#####################################################

pipeline = [
    dict(
        gpu_ids=[0],
        tasks=[
            'skip_plus',
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
            cfg = copy.deepcopy(eval(task))
            cfg = merger.merge(base_cfg, cfg)
            cfg['gpu_ids'] = pipe['gpu_ids']
            cfg['name'] = task
            tasks.append(copy.deepcopy(cfg))

        pipe['tasks'] = tasks

    return pipeline
