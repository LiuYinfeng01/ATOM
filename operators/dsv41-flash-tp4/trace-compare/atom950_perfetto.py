"""Compare ATOM's rocprof and Kineto captures; analyze on CPU with Perfetto v59.1."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from zipfile import ZipFile

from compare_traces import ATOM_AUX, backend, family, read_rank, write_tsv


def unpack(archive, directory):
    directory.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as bundle:
        capture = json.loads(bundle.read("capture.json"))
        validation = json.loads(bundle.read("validation.json"))
        listed = {
            name: digest
            for digest, name in (
                line.split() for line in bundle.read("SHA256SUMS").decode().splitlines()
            )
        }
        assert (
            capture["tp"],
            capture["concurrency"],
            capture["input_tokens"],
            capture["output_tokens"],
        ) == (4, 1, 100000, 1500)
        assert sorted(rank["rank"] for rank in validation["ranks"]) == [0, 1, 2, 3]
        for rank in validation["ranks"]:
            name = f"rank_{rank['rank']}.pftrace"
            path = directory / name
            assert (
                bundle.getinfo(name).file_size == rank["perfetto_bytes"]
                and not rank["perfetto_data_loss"]
            )
            if not path.exists():
                temp = path.with_suffix(".partial")
                with bundle.open(name) as source, temp.open("xb") as target:
                    while block := source.read(4 << 20):
                        target.write(block)
                temp.replace(path)
            with path.open("rb") as source:
                assert hashlib.file_digest(source, "sha256").hexdigest() == listed[name]
            assert rank["forward_calls"] == {"prefill": 14, "decode": 1500}
            print(
                name,
                "SHA verified;",
                rank["gpu_dispatches"]["total"],
                "dispatches in capture receipt",
            )
        return capture, validation, listed


def classify(name, engine):
    sym = name.removesuffix(".kd")
    if sym.startswith("topk_per_row_argmax"):
        return (
            "AITER/FlyDSL",
            "sampling/argmax",
            "flydsl/kernels/topk/topk_per_row_argmax.py",
        )
    if engine == "atom942":
        impl, basis = backend(name, engine)
        return impl, family(name), basis
    special = [
        (
            "mega_mhc_kernel",
            "AITER/HIP",
            "mHC/HIP Mega",
            "mega_mhc.cu; actual module_mega_mhc_stream_h5120 load",
        ),
        (
            "v41_router_s1",
            "ATOM/FlyDSL",
            "MoE/router GEMV",
            "ATOM d6681e153: deepseek_v41/router.py, kernel_symbol and flyc.kernel",
        ),
        (
            "_fused_dual_rmsnorm_mxfp8_quant_kernel",
            "AITER/Triton",
            "QK/dual RMSNorm MXFP8",
            "triton/_triton_kernels/quant/fused_mxfp8_quant.py",
        ),
        (
            "quantize_fp8_kernel",
            "ATOM/Triton",
            "quant/group32 FP8",
            "ATOM d6681e153 model_ops/blockscale_kernels/quantization.py",
        ),
        (
            "_copy_tiles_kernel",
            "ATOM/Triton",
            "cache/copy tiles",
            "model_ops/attentions/pool_layout/paged_state_copy.py",
        ),
        (
            "bmm_a8w8_mxscale_",
            "AITER/FlyDSL",
            "linear/MXscale BMM",
            "flydsl/kernels/bmm_a8w8_mxscale_gfx950.py, name builder",
        ),
        (
            "flydsl_bpreshuffle_mxfp8_",
            "AITER/FlyDSL",
            "linear/MXFP8 preshuffle",
            "explicit generated FlyDSL preshuffle symbol",
        ),
        (
            "flydsl_mxpsh_",
            "AITER/FlyDSL",
            "linear/MXFP8 preshuffle",
            "flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py",
        ),
        (
            "gemm_a16w16_mono_tile_kernel_gfx950",
            "AITER/HIP",
            "linear/Opus BF16",
            "csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_mono_tile_gfx950.cuh",
        ),
        (
            "gemm_a8w8_mxscale_bpreshuffle_wave8_kernel",
            "AITER/HIP",
            "linear/Opus MXFP8",
            "csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_wave8_gfx950.cuh",
        ),
        (
            "moe_reduction_bf16_",
            "AITER/FlyDSL",
            "MoE/reduction",
            "flydsl/kernels/moe_reduce.py, generated name",
        ),
        (
            "gemm2_a4w4_port_",
            "AITER/FlyDSL",
            "MoE/Down",
            "flydsl/kernels/mxfp4_gemm2.py, explicit generated name",
        ),
        (
            "pa_mqa_logits_fp4_rowgroup_kernel",
            "AITER/FlyDSL",
            "index/FP4 scores",
            "flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_rowgroup.py",
        ),
        (
            "radix_topk_one_block_decode_mixed",
            "AITER/FlyDSL",
            "index/radix",
            "flydsl/kernels/topk/radix_topk_one_block.py: mixed/packed/stable name",
        ),
        (
            "radix_topk_one_block_prefill_",
            "AITER/FlyDSL",
            "index/prefill radix",
            "flydsl/kernels/topk/radix_topk_one_block.py",
        ),
        (
            "opus_mla_v4_prefill_",
            "AITER/HIP",
            "attention/Opus main",
            "csrc/kernels/mla/opus/gfx950; name is also used during decode",
        ),
        (
            "pa_decode_ps_reduce_hip_kernel",
            "AITER/HIP",
            "attention/reduce",
            "csrc/cpp_itfs/pa/pa_ps.cuh",
        ),
        (
            "ar_ll_gfx9",
            "AITER/HIP",
            "TP/low-latency sum",
            "csrc/include/custom_all_reduce.cuh, ar_ll_gfx9",
        ),
        (
            "hgemm_bf16_t",
            "AITER/FlyDSL",
            "linear/BF16 GEMM",
            "flydsl/kernels/gemm_a16w16_gfx950.py: hgemm name builder",
        ),
        (
            "bf16gemm_fp32bf16_tn_",
            "AITER/prebuilt .co",
            "linear/BF16 prebuilt",
            "gfx950/bf16gemm/bf16gemm_fp32bf16.csv; producer language not inferred",
        ),
        (
            "inverse_rope_group_quant_kernel",
            "AITER/HIP",
            "output/inverse RoPE+quant",
            "AITER native device symbol; attention._project_out",
        ),
        (
            "rope_hadamard_rotate_activation_fp4quant_kernel",
            "AITER/HIP",
            "index/Q rotate+FP4",
            "native AITER device symbol",
        ),
        (
            "act_and_mul_kernel",
            "AITER/HIP",
            "FFN/act-and-mul",
            "native AITER device template",
        ),
    ]
    for pattern, impl, group, basis in special:
        if pattern in sym:
            return impl, group, basis
    if sym in ATOM_AUX | {"_write_index_fp4_kernel"}:
        return (
            "ATOM/Triton",
            family(name),
            "ATOM model_ops V4.1 device kernels (not the enclosing CPU namespace)",
        )
    impl, basis = backend(name, "atom950")
    return impl, family(name), basis


def inspect_rank(path, receipt, binary):
    from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig

    with TraceProcessor(
        trace=str(path), config=TraceProcessorConfig(bin_path=binary, load_timeout=180)
    ) as tp:

        def query(sql):
            return [vars(row) for row in tp.query(sql)]

        def scalar(sql):
            (row,) = query(sql)
            return next(iter(row.values()))

        query("""CREATE PERFETTO TABLE mark AS
                    SELECT id, ts, dur, name, cast(extract_arg(arg_set_id,'debug.corr_id') AS int) AS cid,
                    CASE WHEN name GLOB 'ATOM::decode*' THEN 'decode' ELSE 'prefill' END AS phase,
                    cast(substr(name,instr(name,'tokens=')+7) AS int) AS tokens,
                    row_number() OVER (PARTITION BY name GLOB 'ATOM::decode*' ORDER BY ts) AS step
                    FROM slice WHERE category='marker_api' AND name GLOB 'ATOM::*'""")
        query("""CREATE PERFETTO TABLE atlas_dispatch AS
                    SELECT id,ts AS display_ts,dur AS display_dur,track_id,name,
                    cast(extract_arg(arg_set_id,'debug.begin_ns') AS int) AS ts,
                    cast(extract_arg(arg_set_id,'debug.end_ns') AS int) AS end_ts,
                    cast(extract_arg(arg_set_id,'debug.delta_ns') AS int) AS dur,
                    cast(extract_arg(arg_set_id,'debug.corr_id') AS int) AS cid,
                    cast(extract_arg(arg_set_id,'debug.queue') AS int) AS queue
                    FROM slice WHERE category='kernel_dispatch'""")
        query(
            "CREATE PERFETTO TABLE joined AS SELECT g.*,m.phase,m.step,m.tokens FROM atlas_dispatch g JOIN mark m USING(cid)"
        )
        tracks = {
            row["id"]: int(re.search(r"\d+", row["name"]).group())
            for row in query(
                "SELECT t.id,t.name FROM track t WHERE t.id IN (SELECT DISTINCT track_id FROM atlas_dispatch)"
            )
        }
        assert scalar("SELECT count(*)-count(DISTINCT cid) FROM mark") == 0
        manifest = receipt["perfetto_slices"]
        for category in ("kernel_dispatch", "marker_api", "memory_copy"):
            assert (
                scalar(f"SELECT count(*) FROM slice WHERE category='{category}'")
                == manifest[category]
            )
        assert (
            scalar(
                "SELECT count(*) FROM atlas_dispatch WHERE ts IS NULL OR end_ts IS NULL OR dur IS NULL OR dur!=end_ts-ts OR dur<0 OR ts!=display_ts"
            )
            == 0
        )
        zero_ns = scalar("SELECT count(*) FROM atlas_dispatch WHERE dur=0")
        times_audit = query(
            "SELECT sum(dur!=display_dur) AS different_durations,sum(display_dur=0) AS zero_display_ns,min(dur) AS min_device_ns,max(abs(dur-display_dur)) AS max_abs_difference_ns FROM atlas_dispatch"
        )[0]
        assert scalar("SELECT count(*) FROM joined") == manifest["kernel_dispatch"]
        assert (
            scalar("SELECT count(*) FROM mark WHERE phase='decode' AND tokens=1")
            == 1500
        )
        assert scalar("SELECT sum(tokens) FROM mark WHERE phase='prefill'") == 100000
        cross = scalar("""SELECT count(*) FROM flow f JOIN slice a ON a.id=f.slice_out JOIN slice b ON b.id=f.slice_in
                    WHERE extract_arg(a.arg_set_id,'debug.corr_id')!=extract_arg(b.arg_set_id,'debug.corr_id')""")
        flow_roots = scalar("""SELECT count(DISTINCT m.cid) FROM flow f JOIN mark m ON m.id=f.slice_out
                    JOIN slice s ON s.id=f.slice_in WHERE s.category IN ('kernel_dispatch','memory_copy')""")
        assert not cross and flow_roots == 1514
        step_rows = query("""SELECT phase,step,tokens,count(*) AS calls, count(DISTINCT name) AS symbols,
                    sum(dur) AS gpu_ns,min(ts) AS start_ts,max(ts+dur)-min(ts) AS device_span_ns
                    FROM joined GROUP BY phase,step,tokens ORDER BY phase,step""")
        assert sum(r["calls"] for r in step_rows) == manifest["kernel_dispatch"]

        def roles(row):
            return "decode" if row["phase"] == "decode" else f"prefill_{row['tokens']}"

        grouped = defaultdict(list)
        for row in step_rows:
            grouped[roles(row)].append(row)
        scopes = {
            name: {
                "steps": len(rows),
                "events": sum(r["calls"] for r in rows),
                "calls_min_max": [
                    min(r["calls"] for r in rows),
                    max(r["calls"] for r in rows),
                ],
                "ms_per_forward": sum(r["gpu_ns"] for r in rows) / (1e6 * len(rows)),
                "device_span_ms_per_forward": sum(r["device_span_ns"] for r in rows)
                / (1e6 * len(rows)),
            }
            for name, rows in grouped.items()
        }
        step10 = next(row for row in grouped["decode"] if row["step"] == 10)
        seq = query(
            "SELECT id,ts,dur,track_id,queue,name FROM joined WHERE phase='decode' AND step=10 ORDER BY ts,id"
        )
        sample = [
            (
                tracks[row["track_id"]],
                row["ts"] - step10["start_ts"],
                row["dur"],
                row["name"],
                row["queue"],
            )
            for row in seq
        ]
        key_stats = query("""SELECT name,count(*) AS calls,sum(dur) AS ns FROM joined WHERE phase='decode'
                    GROUP BY name ORDER BY calls DESC,name""")
        print(
            "rank",
            receipt["rank"],
            "scopes",
            [
                (
                    k,
                    v["steps"],
                    v["events"],
                    v["calls_min_max"],
                    round(v["ms_per_forward"], 4),
                )
                for k, v in scopes.items()
            ],
            "symbols",
            len(key_stats),
            "streams",
            tracks,
            "flow roots",
            flow_roots,
            "zero ns kept",
            zero_ns,
            flush=True,
        )
        if receipt["rank"] == 0:
            print("decode signatures (calls/step, ms/step, name prefix)", flush=True)
            for sym in key_stats:
                print(
                    round(sym["calls"] / 1500, 3),
                    round(sym["ns"] / 1.5e9, 5),
                    sym["name"][:120],
                    flush=True,
                )
            for stream in sorted({item[0] for item in sample}):
                series = [s[3][:90] for s in sample if s[0] == stream]
                print(
                    "step10 stream",
                    stream,
                    "count",
                    len(series),
                    "first/last",
                    series[:13],
                    series[-12:],
                    flush=True,
                )
        counts = query("""SELECT phase,tokens,name,count(*) AS calls,sum(dur) AS ns FROM joined
                    GROUP BY phase, CASE WHEN phase='decode' THEN 1 ELSE tokens END, name ORDER BY phase,tokens,name""")
        for name, rows in grouped.items():
            scopes[name]["median_gpu_ms"] = median(r["gpu_ns"] for r in rows) / 1e6
            scopes[name]["unique_symbols"] = sum(roles(row) == name for row in counts)
        assert all(r["calls"] == 1159 and r["symbols"] == 64 for r in grouped["decode"])
        boundaries = query("""SELECT phase,sum(ts < (SELECT min(ts) FROM mark WHERE phase='decode')) as before,
                    sum(ts >= (SELECT min(ts) FROM mark WHERE phase='decode')) as after FROM joined GROUP BY phase""")
        assert (
            sum(r["before"] for r in boundaries)
            == receipt["gpu_dispatches"]["before_first_decode_marker"]
        )
        assert (
            sum(r["after"] for r in boundaries)
            == receipt["gpu_dispatches"]["from_first_decode_marker"]
        )
        health = query(
            "SELECT name,value FROM stats WHERE severity='error' AND value>0 ORDER BY name"
        )
        rank_summary = {
            "rank": receipt["rank"],
            "scopes": scopes,
            "zero_ns_dispatches_kept": zero_ns,
            "duration_source": "rocprof debug.delta_ns == debug.end_ns - debug.begin_ns; not reconstructed slice.dur",
            "display_time_audit": times_audit,
            "memory_copy_separate": manifest["memory_copy"],
            "zero_cross_id_flows": cross == 0,
            "connected_marker_roots": flow_roots,
            "health": health,
            "wall_clock_cut_audit": boundaries,
        }
        queues = query(
            "SELECT track_id,queue,count(*) AS calls FROM joined WHERE phase='decode' GROUP BY track_id,queue ORDER BY track_id,queue"
        )
        rank_summary["decode_queues"] = [
            {
                "stream": tracks[r["track_id"]],
                "queue": r["queue"],
                "calls_per_step": r["calls"] / 1500,
            }
            for r in queues
        ]
        near = []
        by_queue = []
        windows = []
        if receipt["rank"] == 0:
            query(
                "CREATE PERFETTO TABLE atlas_symbols AS SELECT row_number() OVER(ORDER BY name) AS sid,name FROM (SELECT DISTINCT name FROM joined)"
            )
            query(
                "CREATE PERFETTO TABLE atlas_ordered AS SELECT j.id,j.ts,j.cid,j.phase,j.track_id,j.queue,n.sid FROM joined j JOIN atlas_symbols n USING(name)"
            )
            neighbor_sql = """WITH adj AS (SELECT track_id,queue,phase,cid,sid,
                      lag(sid) OVER w AS prev, lead(sid) OVER w AS next, lag(cid) OVER w AS pc,lead(cid) OVER w AS nc
                                            FROM atlas_ordered WINDOW w AS (PARTITION BY {partition} ORDER BY ts,id))
                                            SELECT {partition},p.name AS prev,m.name AS middle,n.name AS next,count(*) AS hits
                      FROM adj JOIN atlas_symbols p ON p.sid=adj.prev JOIN atlas_symbols m ON m.sid=adj.sid JOIN atlas_symbols n ON n.sid=adj.next
                                            WHERE phase='decode' AND cid=pc AND cid=nc GROUP BY {partition},adj.prev,adj.sid,adj.next ORDER BY hits DESC,{partition},prev,middle,next"""
            near = query(neighbor_sql.format(partition="track_id"))
            by_queue = query(neighbor_sql.format(partition="track_id,queue"))
            windows = query("""WITH phases AS (SELECT *,CASE WHEN phase='prefill' THEN printf('prefill_%02d',step)
                                            WHEN step<=24 THEN 'decode_first24' WHEN step BETWEEN 739 AND 762 THEN 'decode_middle24'
                                            WHEN step>1476 THEN 'decode_last24' END AS win FROM joined)
                                            SELECT win,name,count(*) AS calls,sum(dur) AS ns FROM phases WHERE win IS NOT NULL GROUP BY win,name ORDER BY win,name""")
        return (
            rank_summary,
            step_rows,
            counts,
            sample,
            [
                (tracks[r["track_id"]], r["prev"], r["middle"], r["next"], r["hits"])
                for r in near
            ],
            [
                (
                    tracks[r["track_id"]],
                    r["queue"],
                    r["prev"],
                    r["middle"],
                    r["next"],
                    r["hits"],
                )
                for r in by_queue
            ],
            windows,
        )


def public_outputs(archive, directory, old_traces, out, validation, listed, binary):
    out.mkdir(parents=True, exist_ok=True)
    report = {"atom942_outer": [], "atom950": []}
    inventory = []
    orders = []
    triples = []
    forwards = []
    queue_edges = []
    window_rows = []
    with archive.open("rb") as file:
        report["zip_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
    with ZipFile(archive) as bundle:
        c = json.loads(bundle.read("capture.json"))
        report["atom950_request"] = {
            "runtime": bundle.read("runtime.txt").decode().strip().splitlines(),
            "image": bundle.read("image-id.txt").decode().strip(),
            "traced_request_usage": c["request"]["usage"],
            "traced_http_elapsed_s": c["request"]["elapsed_s"],
            "warmup_only_usage": c["warmup"]["usage"],
            "equal_prompt_fingerprints": c["request"]["input_sha256"]
            == c["warmup"]["input_sha256"],
        }
    for receipt in sorted(validation["ranks"], key=lambda r: r["rank"]):
        rank = receipt["rank"]
        path = directory / f"rank_{rank}.pftrace"
        data, steps, counts, sample, neighbors, queue_neighbors, windows = inspect_rank(
            path, receipt, binary
        )
        data["sha256"] = listed[path.name]
        old, old_symbols, old_order, old_edges = read_rank(
            next((old_traces / f"rank_{rank}").glob("*.json.gz")),
            "atom942",
            rank == 0,
            atom_outer=True,
        )
        old_scope = old["scopes"]["decode"]
        old_data = {
            "rank": rank,
            "sha256": old["sha256"],
            "steps": old["steps"],
            "outer_calls": old_scope["events"],
            "outer_ms_per_step": old_scope["ms_per_step"],
            "outer_unique_symbols": old["unique_symbols_selected"],
        }
        report["atom942_outer"].append(old_data)
        report["atom950"].append(data)
        for row in steps:
            forwards.append(
                [
                    rank,
                    row["phase"],
                    row["step"],
                    row["tokens"],
                    row["calls"],
                    row["symbols"],
                    round(row["gpu_ns"] / 1e6, 6),
                    round(row["device_span_ns"] / 1e6, 6),
                ]
            )
        new_symbols = [
            (
                "decode" if r["phase"] == "decode" else f"prefill_{r['tokens']}",
                r["name"],
                r["calls"],
                r["ns"] / 1000,
            )
            for r in counts
        ]
        for engine, metrics, stats in [
            ("atom950", new_symbols, data),
            ("atom942", [(r[0], r[1], r[5], r[6]) for r in old_symbols], old_data),
        ]:
            groups = defaultdict(lambda: [0, 0.0])
            families = defaultdict(lambda: [0, 0.0])
            distinct = defaultdict(set)
            ids = {
                sym: f"{engine}p-{idx:03d}"
                for idx, sym in enumerate(sorted({r[1] for r in metrics}), 1)
            }
            for scope, sym, calls, us in metrics:
                denom = 24 if engine == "atom942" else data["scopes"][scope]["steps"]
                impl, fam, basis = classify(sym, engine)
                distinct[scope].add(sym)
                for bucket, key in [(groups, impl), (families, fam)]:
                    bucket[scope, key][0] += calls
                    bucket[scope, key][1] += us
                if rank == 0:
                    inventory.append(
                        [
                            engine,
                            scope,
                            ids[sym],
                            impl,
                            fam,
                            calls,
                            denom,
                            round(calls / denom, 6),
                            round(us / (1000 * denom), 6),
                            sym,
                            basis,
                        ]
                    )

            def aggregate(items, current_engine, phase_scopes):
                answer = defaultdict(dict)
                for (scope, key), (num, us) in items.items():
                    denom = (
                        24 if current_engine == "atom942" else phase_scopes[scope]["steps"]
                    )
                    answer[scope][key] = [
                        round(num / denom, 6),
                        round(us / (1000 * denom), 6),
                    ]
                return dict(answer)

            stats["families"] = aggregate(families, engine, data["scopes"])
            stats["backend"] = aggregate(groups, engine, data["scopes"])
            if rank:
                continue
            if engine == "atom950":
                counter = Counter()
                qcounter = Counter()
                for stream, start, dur, sym, queue in sample:
                    counter[stream] += 1
                    qcounter[stream, queue] += 1
                    orders.append(
                        [
                            engine,
                            10,
                            stream,
                            counter[stream],
                            queue,
                            qcounter[stream, queue],
                            start / 1000,
                            dur / 1000,
                            "rocprof marker corr_id (HIP API not recorded)",
                            ids[sym],
                            classify(sym, engine)[1],
                        ]
                    )
                queue_edges.extend(
                    (stream, queue, ids[left], ids[mid], ids[right], n)
                    for stream, queue, left, mid, right, n in queue_neighbors
                )
                for row in windows:
                    denom = 24 if row["win"].startswith("decode") else 1
                    sym = row["name"]
                    window_rows.append(
                        [
                            row["win"],
                            denom,
                            ids[sym],
                            classify(sym, engine)[1],
                            row["calls"],
                            round(row["calls"] / denom, 6),
                            round(row["ns"] / (1e6 * denom), 6),
                        ]
                    )
                edge_src = neighbors
            else:
                first = min(seq[0][0] for seq in old_order.values())
                for stream, seq in sorted(old_order.items()):
                    for ordinal, (start, dur, _, api, sym) in enumerate(seq, 1):
                        orders.append(
                            [
                                engine,
                                10,
                                stream,
                                ordinal,
                                "",
                                "",
                                round(start - first, 3),
                                dur,
                                api,
                                ids[sym],
                                classify(sym, engine)[1],
                            ]
                        )
                edge_src = [
                    (stream, left, mid, right, n)
                    for (stream, _, left, mid, right), n in old_edges.items()
                ]
            for stream, left, mid, right, n in edge_src:
                triples.append([engine, stream, ids[left], ids[mid], ids[right], n])
    write_tsv(
        out / "symbols.tsv",
        [
            "engine",
            "scope",
            "id",
            "implementation",
            "family",
            "event_count",
            "forwards",
            "calls_per_forward",
            "device_ms_per_forward",
            "full_device_symbol",
            "evidence",
        ],
        inventory,
    )
    write_tsv(
        out / "step10-order.tsv",
        [
            "engine",
            "decode_ordinal",
            "stream",
            "ordinal_in_stream",
            "HSA_queue_if_known",
            "ordinal_in_queue",
            "relative_device_start_us",
            "duration_us",
            "host_association",
            "id",
            "family",
        ],
        orders,
    )
    write_tsv(
        out / "neighbors.tsv",
        ["engine", "stream", "previous_id", "current_id", "next_id", "observations"],
        sorted(triples, key=lambda r: (r[0], -r[-1], str(r[1]), r[2:5])),
    )
    write_tsv(
        out / "mi355x-queue-neighbors.tsv",
        ["stream", "HSA_queue", "previous_id", "current_id", "next_id", "observations"],
        queue_edges,
    )
    write_tsv(
        out / "mi355x-windows.tsv",
        [
            "window",
            "forwards",
            "id",
            "family",
            "calls",
            "calls_per_forward",
            "device_ms_per_forward",
        ],
        window_rows,
    )
    write_tsv(
        out / "mi355x-forwards.tsv",
        [
            "rank",
            "phase",
            "ordinal_in_phase",
            "tokens",
            "dispatches",
            "distinct_symbols",
            "GPU_duration_sum_ms",
            "first_to_last_device_span_ms",
        ],
        forwards,
    )
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--unpack-to", type=Path, required=True)
    parser.add_argument("--inspect-rank", type=int, choices=range(4))
    parser.add_argument(
        "--tp-bin", help="Optional path to the Perfetto trace_processor_shell binary"
    )
    parser.add_argument(
        "--atom942", type=Path, help="Directory containing four rank_N/ Kineto traces"
    )
    parser.add_argument(
        "--out", type=Path, help="Publish derived, non-sensitive statistics"
    )
    args = parser.parse_args()
    _, validation, _ = unpack(args.archive, args.unpack_to)
    if args.inspect_rank is not None:
        receipt = next(r for r in validation["ranks"] if r["rank"] == args.inspect_rank)
        inspect_rank(
            args.unpack_to / f"rank_{args.inspect_rank}.pftrace", receipt, args.tp_bin
        )
    if args.out:
        if not args.atom942:
            parser.error("--out requires --atom942")
        with ZipFile(args.archive) as bundle:
            listed = {
                name: digest
                for digest, name in (
                    line.split()
                    for line in bundle.read("SHA256SUMS").decode().splitlines()
                )
            }
        public_outputs(
            args.archive,
            args.unpack_to,
            args.atom942,
            args.out,
            validation,
            listed,
            args.tp_bin,
        )
