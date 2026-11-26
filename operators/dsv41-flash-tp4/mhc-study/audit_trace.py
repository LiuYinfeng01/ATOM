"""Recount the 24 BS1 decode markers in one Kineto rank trace. Duration units: us."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean, median


FAMILIES = {
    'mhc_main_post': 'HAS_POST_1', 'mhc_main_no_post': 'HAS_POST_0',
    'mhc_reduce': '_mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel',
    'mhc_standalone_post': '::mhc_post_kernel', 'expert_gateup': 'moe_2stage_gateup_batch1',
    'expert_down': 'moe_2stage_down_batch1', 'expert_router': 'topk_gating_kernel_opt',
    'tp_reduce': 'lamportAllReduceWaveSplit', 'tp_gather': 'allgather_lastdim',
    'token_quant': 'dynamic_per_token_scaled_quant_kernel', 'group_quant': 'dynamic_per_group_scaled_quant_kernel',
    'qk_norm': 'fused_qk_rmsnorm_group_quant_kernel', 'rmsnorm': 'add_rmsnorm_quant_kernel',
    'rope': 'kn_entry_1c_sbhd_cached_indirect_inplace', 'bf16_gemv': 'wv_splitk_small_fp16_bf16_kernel',
    'index_score': '_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle', 'index_long': 'radix_topk_one_block_kernel',
    'index_compact': 'topk_per_row_decode_1wg_k512_wave64_wv0', 'index_blocks': 'topk_per_row_decode_1wg_k2048_wave64_wv0',
}
EXPECTED = [77, 3, 80, 3, 40, 40, 40, 81, 3, 82, 8, 40, 5, 12, 12, 8, 4, 4, 1]
LAUNCH = ('hipGraphLaunch', 'hipLaunchKernel', 'hipExtLaunchKernel', 'hipModuleLaunchKernel', 'hipExtModuleLaunchKernel', 'hipLaunchCooperativeKernel')


def stats(values):
    return {'mean': mean(values), 'median': median(values), 'min': min(values), 'max': max(values)}


def audit(path):
    data = path.read_bytes()
    trace = json.loads(gzip.decompress(data))
    events = trace['traceEvents']
    marks = sorted((ev for ev in events if ev.get('ph') == 'X' and ev.get('cat') == 'user_annotation' and ev['name'].startswith('decode[bs=1')), key=lambda ev: ev['ts'])
    assert len(marks) == 24
    starts = [mark['ts'] for mark in marks]
    assert all(left['ts'] + left['dur'] <= right['ts'] for left, right in zip(marks, marks[1:]))
    owners = {}
    for ev in events:
        if ev.get('cat') not in ('hip_runtime', 'cuda_driver', 'cuda_runtime') or not ev['name'].startswith(LAUNCH):
            continue
        corr = ev.get('args', {}).get('correlation')
        if corr is None:
            continue
        idx = bisect_right(starts, ev['ts']) - 1
        mark = marks[idx]
        inside = idx >= 0 and (ev['pid'], ev['tid']) == (mark['pid'], mark['tid']) and ev['ts'] < mark['ts'] + mark['dur']
        owners[corr] = idx if corr not in owners and inside else -1
    duration = [Counter() for _ in marks]
    calls = [Counter() for _ in marks]
    symbols = defaultdict(set)
    all_symbols = Counter()
    for ev in events:
        if ev.get('ph') != 'X' or ev.get('cat') != 'kernel' or ev.get('dur', 0) <= 0:
            continue
        idx = owners.get(ev.get('args', {}).get('correlation'), -1)
        if idx < 0:
            continue
        name = ev['name']
        matched = [family for family, tag in FAMILIES.items() if tag in name]
        assert len(matched) <= 1
        family = matched[0] if matched else 'other'
        if matched:
            symbols[family].add(name)
        calls[idx][family] += 1
        duration[idx][family] += ev['dur'] / 1000
        all_symbols[name] += ev['dur'] / (1000 * len(marks))
    for window in calls:
        assert [window[key] for key in FAMILIES] == EXPECTED
    per_step = [dict(step=index + 1, triton_ms=sum(window[key] for key in ('mhc_main_post', 'mhc_main_no_post', 'mhc_reduce')), classified_ms=sum(window[key] for key in FAMILIES), all_attributed_ms=sum(window.values()), total_calls=sum(calls[index].values())) for index, window in enumerate(duration)]
    return {'file': path.name, 'sha256': hashlib.sha256(data).hexdigest(), 'step_count': len(marks),
            'rows': {key: dict(calls_per_step=calls[0][key], ms_per_step=mean(row[key] for row in duration), symbols=sorted(symbols[key])) for key in FAMILIES},
            'per_step': per_step, 'triton_ms': stats([row['triton_ms'] for row in per_step]),
            'classified_ms': mean(row['classified_ms'] for row in per_step), 'all_attributed_ms': mean(row['all_attributed_ms'] for row in per_step),
            'largest_other_ms_per_step': dict(sorted(((name, time) for name, time in all_symbols.items() if not any(tag in name for tag in FAMILIES.values())), key=lambda item: -item[1])[:12])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = audit(args.trace)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('sha256:', result['sha256'], 'decode markers:', result['step_count'])
    for key, info in result['rows'].items():
        print(f"{key:22} {info['calls_per_step']:3} {info['ms_per_step']:.6f} ms")
    print('Triton:', result['triton_ms'], 'classified:', result['classified_ms'], 'total:', result['all_attributed_ms'])