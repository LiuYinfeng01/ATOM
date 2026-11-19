# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2027, Advanced Micro Devices, Inc. All rights reserved.

"""Who sizes the state transfer slots, and who fills them (needs aiter)."""

import types
from unittest import mock

import pytest

gdn_attn = pytest.importorskip("atom.model_ops.attentions.gdn_attn")
model_runner = pytest.importorskip("atom.model_engine.model_runner")


def builder(kv_transfer_config):
    config = types.SimpleNamespace(kv_transfer_config=kv_transfer_config)
    return types.SimpleNamespace(model_runner=types.SimpleNamespace(config=config))


@pytest.mark.parametrize(
    "cfg, slots",
    [
        (
            {
                "kv_connector": "moriio",
                "kv_role": "kv_producer",
                "state_transfer_slots": 8,
            },
            8,
        ),
        ({"kv_connector": "moriio", "state_transfer_slots": 8}, 8),
        (
            {
                "kv_connector": "moriio",
                "kv_role": "kv_consumer",
                "state_transfer_slots": 8,
            },
            0,
        ),
        (
            {
                "kv_connector": "mooncake",
                "kv_role": "kv_producer",
                "state_transfer_slots": 8,
            },
            0,
        ),
        (
            {
                "kv_connector": "multi",
                "connectors": [
                    {
                        "kv_connector": "moriio",
                        "kv_role": "kv_producer",
                        "state_transfer_slots": 8,
                    },
                    {"kv_connector": "lmcache_offload", "kv_role": "offload"},
                ],
            },
            0,
        ),
        ({"kv_connector": "moriio", "kv_role": "kv_producer"}, 0),
        ({}, 0),
    ],
)
def test_only_a_moriio_producer_sizes_transfer_slots(cfg, slots):
    assert gdn_attn.GDNStateMixin.state_transfer_slots(builder(cfg)) == slots


def test_the_runner_copies_staged_state_before_the_connector_sees_it():
    calls = []
    runner = types.SimpleNamespace(
        attn_metadata_builder=types.SimpleNamespace(
            relocate_state_slots=lambda pairs: calls.append(("copy", pairs))
        )
    )
    connector = types.SimpleNamespace(
        start_load_kv=lambda meta: calls.append(("connector", meta))
    )
    meta = types.SimpleNamespace(state_stages={9: (1, 4), 10: (2, 5)})
    with mock.patch.object(model_runner, "get_kvconnector", return_value=connector):
        model_runner.ModelRunner.process_kvconnector_output(runner, meta)
    assert calls == [("copy", [(1, 4), (2, 5)]), ("connector", meta)]
