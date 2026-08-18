"""
Modified from Dassl
"""
import warnings
import torch
import torch.nn as nn

from dassl.optim.radam import RAdam

AVAI_OPTIMS = ["adam", "amsgrad", "sgd", "rmsprop", "radam", "adamw"]


def build_optimizer(model, optim_cfg, param_groups=None):
    optim = optim_cfg.NAME
    lr = optim_cfg.LR
    weight_decay = optim_cfg.WEIGHT_DECAY
    momentum = optim_cfg.MOMENTUM
    sgd_dampening = optim_cfg.SGD_DAMPNING
    sgd_nesterov = optim_cfg.SGD_NESTEROV
    rmsprop_alpha = optim_cfg.RMSPROP_ALPHA
    adam_beta1 = optim_cfg.ADAM_BETA1
    adam_beta2 = optim_cfg.ADAM_BETA2
    staged_lr = optim_cfg.STAGED_LR
    new_layers = optim_cfg.NEW_LAYERS
    base_lr_mult = optim_cfg.BASE_LR_MULT
    new_lr = optim_cfg.NEW_LR
    cat_layers = optim_cfg.CAT_LAYERS
    cat_lr = optim_cfg.CAT_LR
    token_compress_layers = getattr(optim_cfg, 'TOKEN_COMPRESS_LAYERS', [])
    token_compress_lr = getattr(optim_cfg, 'TOKEN_COMPRESS_LR', 0.01)

    if optim not in AVAI_OPTIMS:
        raise ValueError(
            f"optim must be one of {AVAI_OPTIMS}, but got {optim}"
        )

    if param_groups is not None and staged_lr:
        warnings.warn(
            "staged_lr will be ignored, if you need to use staged_lr, "
            "please bind it with param_groups yourself."
        )

    if param_groups is None:
        if staged_lr:
            if not isinstance(model, nn.Module):
                raise TypeError(
                    "When staged_lr is True, model given to "
                    "build_optimizer() must be an instance of nn.Module"
                )

            if isinstance(model, nn.DataParallel):
                model = model.module

            if isinstance(new_layers, str):
                if new_layers is None:
                    warnings.warn("new_layers is empty (staged_lr is useless)")
                new_layers = [new_layers]

            if isinstance(cat_layers, str):
                if cat_layers is None:
                    cat_layers = []
                else:
                    cat_layers = [cat_layers]

            if isinstance(token_compress_layers, str):
                if token_compress_layers is None:
                    token_compress_layers = []
                else:
                    token_compress_layers = [token_compress_layers]

            base_params, new_params, cat_params, token_compress_params = [], [], [], []
            base_modules, new_modules, cat_modules, token_compress_modules = [], [], [], []

            for name, param in model.named_parameters():
                is_token_compress = False
                is_new = False
                is_cat = False

                if 'ddp.ratio_logit' in name or 'ddp.selected_probability' in name:
                    is_token_compress = True
                else:
                    for layer in token_compress_layers:
                        if layer in name:
                            is_token_compress = True
                            break

                if not is_token_compress:
                    for layer in new_layers:
                        if layer in name:
                            is_new = True
                            break

                if not is_token_compress and not is_new:
                    for layer in cat_layers:
                        if layer in name:
                            is_cat = True
                            break

                if is_token_compress:
                    token_compress_modules.append(name)
                    token_compress_params.append(param)
                elif is_new:
                    new_modules.append(name)
                    new_params.append(param)
                elif is_cat:
                    cat_modules.append(name)
                    cat_params.append(param)
                else:
                    base_modules.append(name)
                    base_params.append(param)

            param_groups = []
            group_info = []

            if len(base_params) > 0:
                param_groups.append({
                    "name": "base",
                    "params": base_params,
                    "lr": lr
                })
                group_info.append(("Base", lr, base_modules))

            if len(new_params) > 0:
                param_groups.append({
                    "name": "new",
                    "params": new_params,
                    "lr": new_lr
                })
                group_info.append(("New", new_lr, new_modules))

            if len(cat_params) > 0:
                param_groups.append({
                    "name": "cat",
                    "params": cat_params,
                    "lr": cat_lr
                })
                group_info.append(("CAT", cat_lr, cat_modules))

            if len(token_compress_params) > 0:
                param_groups.append({
                    "name": "token_compress",
                    "params": token_compress_params,
                    "lr": token_compress_lr
                })
                group_info.append(("Token Compression", token_compress_lr, token_compress_modules))

            if len(param_groups) > 1:
                print('Use staged learning rate!')
                for group_name, group_lr, modules in group_info:
                    print(f'{group_name} modules (lr={group_lr}):')
                    if len(modules) > 0:
                        print_string = '\n'.join([f'  - {module}' for module in modules])
                        print(print_string)
                    else:
                        print('  - (no modules in this group)')
            else:
                print('Warning! Use staged learning rate but only one group found, using single learning rate!')
                if isinstance(model, nn.Module):
                    param_groups = model.parameters()
                else:
                    param_groups = model

        else:
            if isinstance(model, nn.Module):
                param_groups = model.parameters()
            else:
                param_groups = model

    if optim == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            eps=1e-4,
        )

    elif optim == "amsgrad":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    elif optim == "radam":
        optimizer = RAdam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )
    else:
        raise NotImplementedError(f"Optimizer {optim} not implemented yet!")

    return optimizer

