import torch
import time
from fvcore.nn import FlopCountAnalysis, flop_count_table

def compute_skiptuning_inference_flops(model, device='cuda', batch_size=1, image_size=224,
                                       show_details=False, max_depth=3):
    model.eval()
    dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(device).type(model.dtype)

    with torch.no_grad():
        start_time = time.time()

        flop = FlopCountAnalysis(model, dummy_input)
        flop.unsupported_ops_warnings(False)
        flop.uncalled_modules_warnings(False)
        flop.tracer_warnings("none")

        total_flops = flop.total()
        flops_per_sample = total_flops / batch_size

        if show_details:
            print(f"\n📊 Detailed FLOPs Breakdown (max_depth={max_depth}):")
            print(flop_count_table(flop, max_depth=max_depth, show_param_shapes=False))

    return {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'batch_size': batch_size,
            'image_size': image_size,
            'device': str(device)
        },
        'inference': {
            'flops_per_sample': flops_per_sample,
            'gflops_per_sample': flops_per_sample / 1e9,
            'analysis_time': time.time() - start_time
        }
    }