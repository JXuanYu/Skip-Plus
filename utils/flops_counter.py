import inspect
import torch
import time
import torch.nn.functional as F


def _build_activities(device):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def _sync_if_cuda(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _infer_batch_size(input_data):
    if torch.is_tensor(input_data):
        return int(input_data.shape[0]) if input_data.ndim > 0 else 1

    if isinstance(input_data, (tuple, list)):
        for item in input_data:
            if torch.is_tensor(item):
                return int(item.shape[0]) if item.ndim > 0 else 1
        return 1

    if isinstance(input_data, dict):
        priority_keys = ["img", "image", "images", "input", "x"]
        for key in priority_keys:
            value = input_data.get(key, None)
            if torch.is_tensor(value):
                return int(value.shape[0]) if value.ndim > 0 else 1
        for value in input_data.values():
            if torch.is_tensor(value):
                return int(value.shape[0]) if value.ndim > 0 else 1

    return 1


def _extract_loss(outputs, labels=None):
    if torch.is_tensor(outputs):
        if outputs.ndim == 0:
            return outputs
        if torch.is_tensor(labels) and labels.ndim > 0 and outputs.ndim >= 2 and outputs.shape[0] == labels.shape[0]:
            return F.cross_entropy(outputs, labels)
        raise TypeError("Tensor output is not a scalar loss and cannot be converted to CE loss.")

    if isinstance(outputs, (tuple, list)):
        # Prefer explicit scalar loss if provided by the model.
        for item in outputs:
            if torch.is_tensor(item) and item.ndim == 0:
                return item

        # Fall back to CE on logits-like tensor.
        if torch.is_tensor(labels) and labels.ndim > 0:
            for item in outputs:
                if torch.is_tensor(item) and item.ndim >= 2 and item.shape[0] == labels.shape[0]:
                    return F.cross_entropy(item, labels)

    raise TypeError(f"Unsupported model output type for FLOPs profiling: {type(outputs)}")


def _is_label_tensor(x):
    return (
        torch.is_tensor(x)
        and x.ndim <= 2
        and x.dtype in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        }
    )


def _split_inputs_and_labels(items):
    if not isinstance(items, (tuple, list)):
        return items, None

    items = list(items)
    if len(items) == 0:
        raise ValueError("Empty parsed batch.")
    if len(items) == 1:
        return items[0], None
    if len(items) == 2:
        return items[0], items[1]

    # Prefer the second item when it already looks like a label tensor
    # (common format: input, label, aux...), otherwise use the last item
    # (common format: input1, input2, ..., label).
    label_idx = 1 if _is_label_tensor(items[1]) else len(items) - 1
    labels = items.pop(label_idx)

    if len(items) == 1:
        return items[0], labels
    return tuple(items), labels


def _forward_for_training_flops(model, model_inputs, labels):
    if isinstance(model_inputs, (tuple, list)):
        if labels is None:
            return model(*model_inputs)
        return model(*model_inputs, labels)

    try:
        forward_sig = inspect.signature(model.forward)
        positional_params = [
            p for p in forward_sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in forward_sig.parameters.values())
        if has_varargs or len(positional_params) >= 2:
            return model(model_inputs, labels)
    except (TypeError, ValueError):
        pass

    return model(model_inputs)


def _prepare_train_batch(batch, device, parse_batch_train_fn):
    if isinstance(batch, (tuple, list)):
        parsed = batch
    else:
        if parse_batch_train_fn is None:
            raise ValueError("parse_batch_train_fn is required when batch is not tuple/list.")
        parsed = parse_batch_train_fn(batch)

    model_inputs, labels = _split_inputs_and_labels(parsed)

    def _to_device(x):
        if torch.is_tensor(x):
            return x.to(device)
        return x

    if isinstance(model_inputs, (tuple, list)):
        model_inputs = tuple(_to_device(x) for x in model_inputs)
    else:
        model_inputs = _to_device(model_inputs)

    labels = _to_device(labels)
    return model_inputs, labels


def _iter_training_batches(feature_loader, train_loader, num_steps):
    if num_steps <= 0:
        return

    if feature_loader is not None:
        source = feature_loader.get_features()
    else:
        source = train_loader

    if source is None:
        return

    for batch_idx, batch in enumerate(source):
        if batch_idx >= num_steps:
            break
        yield batch


def compute_skiptuning_inference_flops(
    model_inference_fn,
    input_batches,
    device="cuda",
    warmup_steps=1,
    show_details=False,
    max_depth=3,
):
    del show_details, max_depth  # 保留参数名，兼容旧调用

    warmup_steps = max(int(warmup_steps), 0)
    profiled_batches = list(input_batches[warmup_steps:])
    profiled_steps = len(profiled_batches)

    if profiled_steps == 0:
        return None

    activities = _build_activities(device)

    start_time = time.time()

    with torch.no_grad():
        for input_data in input_batches[:warmup_steps]:
            _ = model_inference_fn(input_data)

        total_samples = 0
        with torch.profiler.profile(
            activities=activities,
            with_flops=True,
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            for input_data in profiled_batches:
                _ = model_inference_fn(input_data)
                total_samples += _infer_batch_size(input_data)

    _sync_if_cuda(device)

    total_flops = float(sum((evt.flops or 0) for evt in prof.key_averages()))
    flops_per_sample = total_flops / max(total_samples, 1)

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "device": str(device),
            "warmup_steps": warmup_steps,
            "profiled_steps": profiled_steps,
        },
        "inference": {
            "flops_per_sample": flops_per_sample,
            "gflops_per_sample": flops_per_sample / 1e9,
            "analysis_time": time.time() - start_time,
            "samples": total_samples,
            "total_flops": total_flops,
        },
    }


def compute_skiptuning_inference_flops_from_loader(
    model_inference_fn,
    data_loader,
    parse_batch_test_fn,
    device="cuda",
    steps=10,
    warmup_steps=1,
):
    required_batches = int(steps) + max(int(warmup_steps), 0)
    input_batches = []

    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= required_batches:
            break
        parsed = parse_batch_test_fn(batch)
        input_data = parsed[0] if isinstance(parsed, (tuple, list)) else parsed
        input_batches.append(input_data)

    if len(input_batches) <= max(int(warmup_steps), 0):
        return None

    return compute_skiptuning_inference_flops(
        model_inference_fn=model_inference_fn,
        input_batches=input_batches,
        device=device,
        warmup_steps=warmup_steps,
    )


def compute_skiptuning_training_flops(
    model,
    device="cuda",
    steps=10,
    feature_loader=None,
    train_loader=None,
    parse_batch_train_fn=None,
):
    batches = list(_iter_training_batches(feature_loader, train_loader, int(steps)))
    if len(batches) == 0:
        return None

    activities = _build_activities(device)
    model_was_training = model.training
    model.train()

    total_samples = 0
    total_steps = 0
    start_time = time.time()

    try:
        with torch.enable_grad():
            with torch.profiler.profile(
                activities=activities,
                with_flops=True,
                record_shapes=False,
                profile_memory=False,
            ) as prof:
                for batch in batches:
                    model_inputs, labels = _prepare_train_batch(batch, device, parse_batch_train_fn)

                    model.zero_grad(set_to_none=True)
                    outputs = _forward_for_training_flops(model, model_inputs, labels)
                    loss = _extract_loss(outputs, labels)
                    loss.backward()

                    total_steps += 1
                    if torch.is_tensor(labels):
                        total_samples += int(labels.shape[0]) if labels.ndim > 0 else 1
                    else:
                        total_samples += 1

        _sync_if_cuda(device)

        total_flops = float(sum((evt.flops or 0) for evt in prof.key_averages()))
        per_sample_flops = total_flops / max(total_samples, 1)

        return {
            "steps": total_steps,
            "samples": total_samples,
            "total_flops": total_flops,
            "per_sample_flops": per_sample_flops,
            "per_sample_gflops": per_sample_flops / 1e9,
            "analysis_time": time.time() - start_time,
        }
    finally:
        model.zero_grad(set_to_none=True)
        model.train(model_was_training)
