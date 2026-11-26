"""Compare two specific C1 captures using unique HIP launch correlations. CPU only."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re


LAUNCH = ('hipGraphLaunch', 'hipLaunchKernel', 'hipModuleLaunchKernel', 'hipExtModuleLaunchKernel', 'hipExtLaunchKernel', 'hipLaunchCooperativeKernel', 'hipMemcpy', 'hipMemset')
RUNTIME = {'cuda_runtime', 'cuda_driver', 'hip_runtime'}
VLLM_TRITON = {
    'model_executor/kernels/mhc/triton.py': '_mhc_pre_mix_kernel _hc_head_reduce_store_kernel',
    'model_executor/kernels/linear/mxfp8/rocm_native.py': '_mxfp8_linear_kernel',
    'v1/attention/ops/rocm_aiter_mla_sparse.py': '_sparse_attn_decode_gfx950_partial_kernel _sparse_attn_decode_reduce_kernel _inverse_rope_gptj_kernel _mask_candidates_strided_kernel',
    'models/deepseek_v41/amd/rocm.py': '_compute_topk_lens_kernel _pack_global_topk_ragged_kernel',
    'model_executor/kernels/attention/dsa/candidate_blocks.py': '_candidate_flags_kernel _block_scores_kernel _store_candidates_kernel',
    'models/deepseek_v41/common/engram.py': '_engram_lookup_kernel _fused_engram_post_wkv_kernel _hash_ids_kernel',
    'models/deepseek_v41/common/ops/fused_compress_quant_cache.py': '_fused_save_compress_norm_kernel _rope_quant_insert_kernel',
    'models/deepseek_v41/common/ops/indexer_k_store.py': '_indexer_k_norm_rope_quant_store_kernel',
    'models/deepseek_v41/compressor.py': '_ring_slot_mapping_kernel',
    'v1/worker/gpu/block_table.py': '_gather_block_tables_kernel _compute_slot_mappings_kernel',
    'v1/worker/gpu/buffer_utils.py': '_apply_write_kernel', 'v1/worker/utils.py': '_zero_kv_blocks_kernel',
    'v1/worker/gpu/input_batch.py': '_combine_sampled_and_draft_tokens_kernel _expand_idx_mapping_kernel _get_num_sampled_and_rejected_kernel _post_update_kernel _prepare_pos_seq_lens_kernel',
    'v1/worker/gpu/spec_decode/dflash/speculator.py': '_prepare_dflash_inputs_kernel',
    'v1/worker/gpu/spec_decode/rejection_sampler_utils.py': '_compute_local_logits_stats_kernel _insert_resampled_kernel _rejection_kernel _resample_kernel',
    'v1/worker/gpu/sample/gumbel.py': '_gumbel_sample_kernel',
    'v1/attention/ops/metadata.py': '_token_request_mapping_kernel',
    'v1/attention/backends/mla/sparse_swa.py': '_compute_swa_indices_and_lens_kernel',
    'v1 model-state lookback helper (same-name triton.jit)': '_gather_lookback_kernel',
}
VDEFS = {sym: path for path, names in VLLM_TRITON.items() for sym in names.split()}
ATOM_AUX = set('''_sparse_attn_v4_paged_2src_kernel _sparse_attn_v4_paged_2src_csa_kernel _sparse_attn_v4_paged_2src_csa_split_kernel _sparse_attn_v4_split_combine_kernel _engram_hash_kernel _engram_cursor_kernel _engram_snapshot_kernel _compress_kernel _engram_post_wkv_kernel _uva_lookup_kernel _masked_embedding_kernel _rope_window_kernel _inverse_rope_gptj_kernel _indices _indptr_scan_all _window_rows _write_index_kernel _unit_table_kernel _scale_indexer_weights_kernel _block_maxima_kernel _candidate_table_kernel _lift_kernel _update_compressor_states_kernel _swa_write_kernel _fused_compress_attn_kernel _collapse_kernel _gather_rows _fill_deferred_decode_ids_kernel scatter_packed_bytes'''.split())


def backend(name, engine):
    sym = name.removesuffix('.kd')
    if sym.startswith('_gluon_deepgemm_'):
        return 'AITER/Gluon', 'pa_mqa_logits.py: gluon.jit'
    if sym.startswith(('moe_2stage_', 'mfma_moe1_', 'mfma_moe2_', 'topk_per_row_decode_1wg_')):
        return 'AITER/FlyDSL', 'gfx9_fp8 batch1, mixed_moe_gemm_2stage name builder, or topk_per_row_decode_persistent'
    if sym.startswith('_mhc_fused_post_pre_delayed_'):
        return 'AITER/Triton', 'mhc_fused_post_pre_delayed_rmsnorm'
    if sym in ATOM_AUX and engine == 'atom942':
        return 'ATOM/Triton', 'ATOM device implementation in 5f6bdb75; not CPU namespace'
    if sym in VDEFS and engine == 'vllm950':
        return 'vLLM/Triton', VDEFS[sym] + ' (corresponding definition, not proof of exact build)'
    if sym.startswith('triton_poi_'):
        return 'Inductor/Triton', 'explicit generated Triton symbol'
    if 'tilelang_kernel' in sym:
        return 'TileLang', 'explicit generated TileLang symbol'
    if sym.startswith(('void aiter::', '_ZN5aiter')):
        return 'AITER/HIP', 'named native AITER device template'
    if sym.startswith(('void vllm::', 'void wvSplitK_hf_sml_')):
        return 'vLLM/HIP', 'named vLLM native device template or csrc/rocm skinny GEMV'
    if sym.startswith('Cijk_') or sym.startswith('Cijk'):
        return 'ROCm BLAS', 'Tensile-style device symbol; not a unique projection'
    if sym.startswith('void at::'):
        return 'PyTorch/ATen', 'ATen device template'
    if sym.startswith('void rocprim::'):
        return 'rocPRIM', 'rocPRIM device primitive'
    if sym.startswith('ncclDevKernel'):
        return 'RCCL', 'collective device symbol on ROCm'
    if sym.startswith('__amd_rocclr_'):
        return 'ROCm runtime', 'runtime copy/fill kernel'
    return 'Unresolved', 'generic symbol or no verified generator; .kd is not an implementation language'


def family(name):
    keys = [('mhc_fused_post_pre_delayed_rmsnorm_main','mHC/Triton main'),('mhc_fused_post_pre_delayed_rmsnorm_reduce','mHC/Triton reduce'),
            ('mhc_fused_post_pre_gemm_sqrsum','mHC/HIP fused GEMM+sqrsum'),('mhc_pre_gemm_sqrsum','mHC/HIP pre GEMM+sqrsum'),('mhc_pre_big_fuse_tilelang','mHC/TileLang gates'),('hc_prenorm_gemm_tilelang','mHC/TileLang prenorm GEMM'),('mhc_pre_big_fuse_kernel','mHC/HIP gates'),('mhc_pre_mix_kernel','mHC/Triton pre mix'),('hc_head_reduce_store','mHC/Triton head reduce'),('mhc_post_kernel','mHC/separate post'),
            ('gluon_deepgemm_fp8_paged_mqa_logits','index/FP8 scores'),('radix_topk_one_block_kernel','index/radix'),('topk_per_row_decode_1wg_k512','index/compact512'),('topk_per_row_decode_1wg_k2048','index/block2048'),
            ('topKPerRowDecode512DeviceLengthAware<1024, false>','index/Top512 split'),('topKPerRowDecode512DeviceLengthAware<1024, true>','index/Top512 merge'),
            ('sparse_attn_v4_paged_2src_csa_split','attention/partial'),('sparse_attn_decode_gfx950_partial','attention/partial'),('sparse_attn_v4_split_combine','attention/reduce'),('sparse_attn_decode_reduce','attention/reduce'),('sparse_attn_v4_paged_2src','attention/unsplit'),
            ('moe_2stage_gateup','MoE/GateUp'),('mfma_moe1_','MoE/GateUp'),('moe_2stage_down','MoE/Down'),('mfma_moe2_','MoE/Down'),('topk_gating_kernel_opt','MoE/router'),('topkGatingSoftplusSqrt','MoE/router'),('opus_moe_sorting_entry','MoE/sort'),('fused_mx_quant_moe_sort','MoE/quant sort'),
            ('lamportAllReduce','TP/Lamport sum'),('cross_device_reduce_1stage','TP/one-stage sum'),('allgather_lastdim','TP/allgather'),('ncclDevKernel','TP/RCCL'),
            ('dynamic_per_token_scaled_quant','quant/per-token'),('dynamic_per_group_scaled_quant','quant/per-group'),('fused_qk_rmsnorm_group_quant','QK/RMSNorm quant'),('fusedDeepseekV4QNormRopeKVRopeQuantInsert','QK/fused norm RoPE KV'),('add_rmsnorm_quant','RMSNorm/template'),('_rope_window_kernel','KV/RoPE window'),('_inverse_rope_gptj','output/inverse RoPE'),('_mxfp8_linear_kernel','linear/MXFP8'),('Cijk_','linear/BLAS')]
    return next((label for match, label in keys if match in name), name.removesuffix('.kd') if len(name)<72 else backend(name, '')[0])


def write_tsv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t', lineterminator='\n'); writer.writerow(fields); writer.writerows(rows)


def read_rank(path, engine, detailed=False):
    with path.open('rb') as file:
        sha = hashlib.file_digest(file, 'sha256').hexdigest()
    with gzip.open(path,'rb') as file:
        doc = json.load(file)
    events=doc['traceEvents']; vllm=engine=='vllm950'
    marks = sorted((ev for ev in events if ev.get('cat')=='user_annotation' and ev.get('ph')=='X' and (ev['name']=='execute_context_0(0)_generation_1(6)' if vllm else ev['name']=='decode[bs=1 tok=1 d=1]')), key=lambda ev:ev['ts'])
    assert len(marks)==(20 if vllm else 24) and all(left['ts']+left['dur']<=right['ts'] for left,right in zip(marks,marks[1:]))
    starts=[ev['ts'] for ev in marks]; captures=Counter(); owners={}; graph_ancestry=Counter(); cpu=defaultdict(list)
    if vllm:
        for ev in events:
            if ev.get('cat')=='python_function' and ev.get('ph')=='X':
                if re.search(r'model_runner.py\(\d+\): execute_model$',ev['name']): cpu['execute'].append(ev)
                if re.search(r'speculator.py\(\d+\): propose$',ev['name']):cpu['propose'].append(ev)
                if detailed and re.search(r'triton/runtime/jit.py\(\d+\): run$',ev['name']):cpu['triton'].append(ev)
    for ev in events:
        if ev.get('cat') not in RUNTIME or ev.get('ph')!='X' or not ev['name'].startswith(LAUNCH): continue
        corr=ev.get('args',{}).get('correlation');
        if corr is None: continue
        if corr in owners: owners[corr]=None;continue
        idx=bisect_right(starts,ev['ts'])-1;mark=marks[idx]
        inside=idx>=0 and ev['pid']==mark['pid'] and ev['tid']==mark['tid'] and ev['ts']<mark['ts']+mark['dur']
        graph=ev['name']=='hipGraphLaunch'
        if vllm and graph:
            parent=[key for key,rows in cpu.items() if any(p['pid']==ev['pid'] and p['tid']==ev['tid'] and p['ts']<=ev['ts']<p['ts']+p['dur'] for p in rows)]
            assert parent==(['execute'] if inside else ['propose']); graph_ancestry[parent[0]]+=1
        role=('target_graph' if graph else 'target_aux') if inside else ('draft_graph' if graph else 'outside_aux')
        if not vllm: role='decode' if inside else 'excluded'
        traced_jit = vllm and detailed and not graph and any(p['pid']==ev['pid'] and p['tid']==ev['tid'] and p['ts']<=ev['ts']<p['ts']+p['dur'] for p in cpu['triton'])
        owners[corr]=(idx,role,ev['name'],traced_jit)
    if vllm: assert graph_ancestry=={'execute':20,'propose':20}
    counters=defaultdict(Counter);times=defaultdict(Counter);perstep=defaultdict(lambda:[0]*len(marks));order=defaultdict(list);totals=Counter();whole_symbols=Counter()
    kernels=sorted((ev for ev in events if ev.get('cat')=='kernel' and ev.get('ph')=='X' and ev.get('dur',0)>0),key=lambda ev:ev['ts'])
    stream_rows=defaultdict(list); phases=Counter();incount=0; traced_generic=Counter()
    for ev in kernels:
        ref=owners.get(ev['args'].get('correlation')); assert ref is not None, 'missing or ambiguous HIP correlation'
        idx,role,launch,traced_jit=ref; stream=ev['args'].get('stream',ev['tid']);symbol=ev['name'];dur=ev['dur']
        if role=='excluded':
            if detailed:stream_rows[stream].append(None)
            continue
        assert 0<=idx<len(marks)
        incount+=1;whole_symbols[symbol]+=1; phases[role]+=1
        counters[role][symbol]+=1;times[role][symbol]+=dur;perstep[role][idx]+=1;totals[role]+=dur
        if traced_jit:traced_generic[role,symbol]+=1
        if detailed:
            stream_rows[stream].append((idx,role,symbol))
            if idx==9:order[stream].append((ev['ts'],dur,role,launch,symbol))
    assert len(kernels)==incount if vllm else incount==24*967
    if vllm:
        assert perstep['target_graph']==[2225]*20 and perstep['draft_graph']==[204]*20 and perstep['outside_aux']==[73]*20
        assert sum(perstep['target_aux'])==2328 and incount==52368
    else: assert perstep['decode']==[967]*24
    groups=defaultdict(lambda:[0,0.]);families=defaultdict(lambda:[0,0.]);symbol_rows=[]
    for role,counts in counters.items():
        for sym,count in counts.items():
            impl,proof=backend(sym,engine); fam=family(sym);us=times[role][sym];
            if vllm and detailed and impl=='Unresolved' and traced_generic[role,sym]==count:
                impl,proof='vLLM/Triton','all eager launches in THIS scope inside recorded Triton JIT; no extrapolation to graph replay'
            for label in [role,'window']+(['target'] if vllm and role.startswith('target_') else []):
                for mapping,key in ((groups,(label,impl)),(families,(label,fam))):
                    mapping[key][0]+=count;mapping[key][1]+=us
            symbol_rows.append((role,sym,impl,fam,proof,count,us))
    def shape_stats(d):
        result=defaultdict(dict)
        for (role,key),(count,us) in d.items():result[role][key]=[count/len(marks),round(us/(1000*len(marks)),6)]
        return dict(result)
    edges=Counter()
    if detailed:
        for stream,seq in stream_rows.items():
            for left,mid,right in zip(seq,seq[1:],seq[2:]):
                if left and mid and right and left[0]==mid[0]==right[0]:edges[(str(stream),mid[1],left[2],mid[2],right[2])]+=1
    dev=doc['deviceProperties'][0]
    result={'sha256':sha,'device':dev['name'],'CU':dev['numSms'],'marker':marks[0]['name'],'steps':len(marks),'kernel_events_file':len(kernels),'selected_events':incount,'unique_symbols_selected':len(whole_symbols),'scopes':{s:{'events':n,'calls_per_step':n/len(marks),'ms_per_step':round(totals[s]/(1000*len(marks)),6),'per_step_counts':perstep[s]} for s,n in phases.items()},'all_selected_ms_per_step':round(sum(totals.values())/(1000*len(marks)),6),'graph_parent_counts':dict(graph_ancestry)}
    if detailed: result.update(backend=shape_stats(groups),families=shape_stats(families))
    return result,symbol_rows,order,edges


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atom-trace-dir',type=Path,required=True,help='Parent of rank_0..rank_3 for the 24-step gfx942 capture')
    parser.add_argument('--vllm-dir',type=Path,required=True,help='Extracted archive root containing c1 and capture_validation.json')
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    validation=json.loads((args.vllm_dir/'capture_validation.json').read_text()); receipts=next(x for x in validation['points'] if x['concurrency']==1)['traces']
    rows=[];order_rows=[];edge_rows=[];report={}
    for engine in ['atom942','vllm950']:
        report[engine]=[]
        for rank in range(4):
            files=list((args.vllm_dir/'c1/profiles').glob(f'*rank{rank}.*.json.gz')) if engine=='vllm950' else list((args.atom_trace_dir/f'rank_{rank}').glob('*.json.gz'))
            assert len(files)==1
            stats,symbols,order,edges=read_rank(files[0],engine,rank==0);report[engine].append(stats)
            if engine=='vllm950':
                receipt=next(r for r in receipts if Path(r['path']).name==files[0].name);assert receipt['sha256']==stats['sha256'] and receipt['gpu_kernel_events']==stats['kernel_events_file']
            if rank:continue
            ids={sym:f'{engine}-{idx:03d}' for idx,sym in enumerate(sorted({r[1] for r in symbols}),1)}
            for role,sym,impl,fam,proof,count,us in sorted(symbols):
                rows.append((engine,role,ids[sym],impl,fam,count,count/stats['steps'],round(us/(1000*stats['steps']),6),sym,proof))
            first=min(seq[0][0] for seq in order.values())
            for stream,seq in sorted(order.items()):
                for ordinal,(start,dur,role,launch,sym) in enumerate(seq,1):
                    order_rows.append((engine,10,stream,ordinal,round(start-first,3),dur,role,launch,ids[sym],family(sym)))
            for (stream,role,left,middle,right),n in edges.most_common():edge_rows.append((engine,stream,role,ids[left],ids[middle],ids[right],n))
    write_tsv(args.out/'symbols.tsv',['engine','scope','symbol_id','implementation','op_family','window_count','calls_per_iteration','GPU_ms_per_iteration','full_symbol','basis'],rows)
    write_tsv(args.out/'step10-order.tsv',['engine','iteration_1based','stream','stream_ordinal','GPU_start_us_relative','duration_us','scope','host_launch_API','symbol_id','op_family'],order_rows)
    write_tsv(args.out/'neighbors.tsv',['engine','stream','middle_scope','previous_symbol_id','symbol_id','next_symbol_id','observations'],edge_rows)
    (args.out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    for engine,ranks in report.items():
        first=ranks[0];print(engine,'steps',first['steps'],'rank0 unique',first['unique_symbols_selected'],'ms/iteration',first['all_selected_ms_per_step'],{key:row['calls_per_step'] for key,row in first['scopes'].items()})


if __name__=='__main__':
    main()