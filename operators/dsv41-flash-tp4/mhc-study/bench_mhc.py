"""Offline gfx942 A/B: real checkpoint parameters, synthetic activations, no model server."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from random import Random
from statistics import median
from unittest.mock import patch

import torch
from safetensors import safe_open
from aiter.ops.mega_mhc import mega_mhc, mega_mhc_shuffle_fn, prepare_mega_mhc
from aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm import mhc_fused_post_pre_delayed_rmsnorm


OUTPUTS = ('residual', 'post', 'comb', 'normed', 'next_pre')
HIDDEN = 5120
EPS = 1e-20
GATE_EPS = 1e-6


def load_params(folder, smoke):
    mapping = json.loads((folder / 'model.safetensors.index.json').read_text())['weight_map']
    result, shards = [], {}
    with ExitStack() as stack:
        def load(key):
            name = mapping[key]
            if name not in shards:
                shards[name] = stack.enter_context(safe_open(str(folder / name), framework='pt', device='cpu'))
            return shards[name].get_tensor(key).to('cuda')
        for layer in range(1 if smoke else 40):
            for sublayer in ('attn', 'ffn'):
                prefix = f'layers.{layer}.'
                coeff = {part: load(prefix + f'hc_{sublayer}_{part}') for part in ('fn', 'scale', 'base')}
                coeff['norm_weight'] = load(prefix + sublayer + '_norm.weight')
                assert coeff['fn'].shape == (24, 20480) and coeff['fn'].dtype == torch.float32
                assert coeff['scale'].shape == (3,) and coeff['base'].shape == (24,)
                assert coeff['norm_weight'].shape == (HIDDEN,) and coeff['norm_weight'].dtype == torch.bfloat16
                coeff['packed'] = mega_mhc_shuffle_fn(coeff['fn'])
                result.append((layer, sublayer, coeff))
    return result


def make_case(entry, seed):
    layer, sublayer, weights = entry
    gen = torch.Generator(device='cuda').manual_seed(seed + 2 * layer + int(sublayer == 'ffn'))
    def normal(*shape):
        return torch.randn(*shape, generator=gen, device='cuda')
    residual = normal(1, 4, HIDDEN).bfloat16()
    incoming = torch.sigmoid(normal(1, 4)) + GATE_EPS
    if layer == 0 and sublayer == 'attn':
        incoming.zero_(); incoming[:, 0] = 1
    extras = {'norm_weight': weights['norm_weight'], 'norm_eps': EPS}
    if sublayer != 'attn' or layer not in (0, 1, 14):
        extras.update(sublayer_out=normal(1, HIDDEN).bfloat16(), post_layer_mix=torch.sigmoid(normal(1, 4)) * 2, comb_res_mix=torch.softmax(normal(1, 4, 4), dim=-1))
    return residual, incoming, weights, extras


def run_op(case, workspace=None):
    residual, incoming, weights, extras = case
    args = (weights['scale'], weights['base'], EPS, GATE_EPS, GATE_EPS, 2., 20, incoming)
    if workspace is None:
        return mhc_fused_post_pre_delayed_rmsnorm(residual, weights['fn'], *args, **extras)
    return mega_mhc(residual, weights['packed'], *args, **extras, workspace=workspace, w_preshuffle_bf16=True, out_dtype='bf16')


def reference(case):
    residual, incoming, weights, extra = case
    if 'sublayer_out' in extra:
        carried = extra['comb_res_mix'].transpose(1, 2).double() @ residual.double()
        residual = (carried + extra['post_layer_mix'].double()[:, :, None] * extra['sublayer_out'].double()[:, None, :]).bfloat16()
    flat = residual.double().flatten(1)
    pred = (flat @ weights['fn'].double().T) * torch.rsqrt(flat.square().mean(1, keepdim=True) + EPS)
    scale, bias = weights['scale'].double(), weights['base'].double()
    pre = torch.sigmoid(pred[:, :4] * scale[0] + bias[:4]) + GATE_EPS
    post = 2 * torch.sigmoid(pred[:, 4:8] * scale[1] + bias[4:8])
    comb = (pred[:, 8:] * scale[2] + bias[8:]).reshape(1, 4, 4).softmax(-1) + GATE_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + GATE_EPS)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + GATE_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + GATE_EPS)
    collapsed = (residual.double() * incoming.double()[:, :, None]).sum(1).bfloat16().double()
    normed = (collapsed * torch.rsqrt(collapsed.square().mean(1, keepdim=True) + EPS) * weights['norm_weight'].double()).bfloat16()
    return residual, post.float().unsqueeze(-1), comb.float(), normed, pre.float()


def compare(record, outputs, expected):
    for name, actual, ref in zip(OUTPUTS, outputs, expected, strict=True):
        assert actual.shape == ref.shape and actual.dtype == ref.dtype
        actual, ref = actual.double(), ref.double()
        assert bool(torch.isfinite(actual).all() & torch.isfinite(ref).all())
        atol, rtol = (0.02, 0.01) if name in ('residual', 'normed') else (0.0005, 0.001)
        error = (actual - ref).abs()
        item = record.setdefault(name, {'calls': 0, 'elements': 0, 'outside_tolerance': 0, 'max_abs': 0., 'max_rel_l2': 0.})
        item['calls'] += 1; item['elements'] += error.numel()
        item['outside_tolerance'] += int((error > atol + rtol * ref.abs()).sum())
        item['max_abs'] = max(item['max_abs'], float(error.max()))
        item['max_rel_l2'] = max(item['max_rel_l2'], float(error.norm() / ref.norm().clamp_min(1e-20)))


def graph_of(cases, workspace, stream):
    for case in (cases[0], cases[-1]):
        run_op(case, workspace)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [run_op(case, workspace) for case in cases]
    graph.replay(); stream.synchronize()
    return graph, outputs


def time_graph(bundle, repeats):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        bundle[0].replay()
    end.record(); end.synchronize()
    return start.elapsed_time(end) / repeats


def abba(baseline, candidate, rounds=9, repeats=8):
    pairs = []
    for index in range(rounds):
        order = ('A', 'B', 'B', 'A') if index % 2 == 0 else ('B', 'A', 'A', 'B')
        blocks = {'A': [], 'B': []}
        for tag in order:
            blocks[tag].append(time_graph(baseline if tag == 'A' else candidate, repeats))
        pairs.append([median(blocks['A']), median(blocks['B'])])
    return {'pairs_ms_per_graph': pairs, 'A_triton_ms': median(pair[0] for pair in pairs), 'B_hip_ms': median(pair[1] for pair in pairs), 'paired_A_minus_B_ms': median(pair[0] - pair[1] for pair in pairs)}


def symbols(case, workspace):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        run_op(case, workspace); torch.cuda.synchronize()
    return sorted({event.name for event in profiler.events() if getattr(event.device_type, 'name', '') == 'CUDA' and event.name and 'memcpy' not in event.name.lower()})


@torch.inference_mode()
def bench(model, smoke, include_split):
    assert torch.cuda.device_count() == 1
    prop = torch.cuda.get_device_properties(0)
    assert prop.gcnArchName.split(':')[0] == 'gfx942' and prop.multi_processor_count == 80
    dev = torch.device('cuda', 0)
    bank = load_params(model, smoke)
    cases = [make_case(entry, 41000) for entry in bank]
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    result = {'scope': 'one MI308X; M=1; real checkpoint mHC weights; fixed synthetic activations, not a trace tensor dump or server run', 'torch': str(torch.__version__), 'hip': torch.version.hip, 'gpu': prop.name, 'cu': prop.multi_processor_count, 'seeds': [41000, 42000], 'weights': len(bank), 'correctness': {}, 'image_src_sha256': {name: hashlib.sha256(Path('/app/aiter-test', name).read_bytes()).hexdigest() for name in ('aiter/ops/mega_mhc.py', 'csrc/kernels/mega_mhc.cu')}}
    with torch.cuda.stream(stream):
        from atom.model_ops.deepseek_v41 import mhc_pre_delayed as dispatch
        with patch.object(dispatch.envs, 'ATOM_DSV41_MHC_BACKEND', 'auto'):
            assert dispatch.pack_mhc_fn(bank[0][2]['fn']) is None
        with patch.object(dispatch.envs, 'ATOM_DSV41_MHC_BACKEND', 'hip'):
            try:
                dispatch.pack_mhc_fn(bank[0][2]['fn'])
            except ValueError as exc:
                result['stock_atom_forced_hip'] = str(exc)
            else:
                raise AssertionError('Stock ATOM unexpectedly accepted gfx942')
        def meta(tensor):
            return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}
        residual, incoming, weights, extras = cases[1]
        result['measured_inputs'] = {name: meta(tensor) for name, tensor in {'residual': residual, 'incoming_pre': incoming, **weights, **{name: value for name, value in extras.items() if torch.is_tensor(value)}}.items()}
        auto = prepare_mega_mhc(dev, 1, HIDDEN)
        assert auto.variant == 'generic'
        result['auto_split'] = auto.split_k
        print('Device', prop.name, 'weights', len(bank), 'generic split', auto.split_k, flush=True)
        plans = {auto.split_k: auto}
        rotated_a = graph_of(cases, None, stream)
        rotated_auto = graph_of(cases, auto, stream)
        result['dispatch'] = {f'{tag}_{mode}': symbols(cases[idx], ws) for idx, mode in [(0, 'no_post'), (1, 'post')] for tag, ws in [('triton', None), ('hip', auto)]}
        result['measured_outputs'] = {name: meta(value) for name, value in zip(OUTPUTS, run_op(cases[1], auto), strict=True)}
        print('Dispatch', result['dispatch'], flush=True)
        if not smoke:
            for split in (1, 2, 4, 5, 8, 10, 16, 32):
                plans[split] = prepare_mega_mhc(dev, 1, HIDDEN, split_k=split)
        graphs = {auto.split_k: rotated_auto}
        for split, plan in plans.items():
            if split not in graphs:
                graphs[split] = graph_of(cases, plan, stream)
        screening = {split: [] for split in plans}
        rng = Random(1729)
        for _ in range(5):
            order = list(plans); rng.shuffle(order)
            for split in order:
                screening[split].append(time_graph(graphs[split], 6))
        best = min(screening, key=lambda split: median(screening[split]))
        result['screening_ms_per_graph'] = screening; result['best_split'] = best
        held = sorted({best, auto.split_k, *include_split})
        for seed in result['seeds']:
            for entry in bank:
                case = make_case(entry, seed); ref = reference(case); triton_out = run_op(case)
                compare(result['correctness'].setdefault('triton_vs_ref', {}), triton_out, ref)
                for split in held:
                    hip_out = run_op(case, plans[split]); compare(result['correctness'].setdefault(f'hip{split}_vs_ref', {}), hip_out, ref)
                    compare(result['correctness'].setdefault(f'hip{split}_vs_triton', {}), hip_out, triton_out)
        for label, bundle in [('triton', rotated_a), *((f'hip{split}', graphs[split]) for split in held)]:
            bundle[0].replay(); bundle[0].replay(); stream.synchronize()
            for case, outputs in zip(cases, bundle[1], strict=True):
                compare(result['correctness'].setdefault(label + '_replayed_vs_ref', {}), outputs, reference(case))
        result['all_within_tolerance'] = all(not val['outside_tolerance'] for metrics in result['correctness'].values() for val in metrics.values())
        print('Numerics', result['all_within_tolerance'], 'best split', best, flush=True)
        for split in held:
            result[f'rotate_split{split}'] = abba(rotated_a, graphs[split])
            if not smoke:
                for idx, label in [(0, 'no_post'), (1, 'post')]:
                    hot = [cases[idx]] * 32
                    result[f'hot32_{label}_split{split}'] = abba(graph_of(hot, None, stream), graph_of(hot, plans[split], stream))
        result['finished'] = True
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path); parser.add_argument('output', type=Path); parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--include-split', action='append', type=int, default=[])
    args = parser.parse_args()
    result = bench(args.model, args.smoke, args.include_split)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('FINISHED', {key: val['paired_A_minus_B_ms'] for key, val in result.items() if key.startswith('rotate_')}, flush=True)