# SPDX-License-Identifier: MIT
"""Scheduler renewal tickets use actual futures, with the native RPC controlled."""

from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload.mooncake_store.renewal_gate import (
    RenewalGate,
    RenewalState,
)


class _Executor:
    def __init__(self):
        self.jobs = []

    def submit(self, fn, keys):
        future = Future()
        self.jobs.append((future, fn, keys))
        return future

    def finish(self, index, result=0):
        future = self.jobs[index][0]
        if future.set_running_or_notify_cancel():
            future.set_result(result)


@pytest.fixture
def gate():
    clock = SimpleNamespace(now=0.0)
    executor = _Executor()
    client = SimpleNamespace(
        capacity=2,
        renew_executor=executor,
        backend_slot=SimpleNamespace(take=lambda **kw: nullcontext()),
        backend=SimpleNamespace(renew_keys=lambda keys: 0),
    )
    return RenewalGate(client, clock=lambda: clock.now), executor, clock


def test_pending_renewal_is_submitted_once_and_zero_hit_still_dispatches(gate):
    ticket_gate, executor, _ = gate
    owner = object()
    for _ in range(3):
        assert (
            ticket_gate.request(owner, "a", ("key",), (0, 256)) is RenewalState.PENDING
        )
    assert len(executor.jobs) == 1
    future, fn, keys = executor.jobs[0]
    assert keys == ("key",)  # No Sequence reaches the executor.
    future.set_result(fn(keys))
    assert ticket_gate.request(owner, "a", keys, (0, 256)) is RenewalState.READY
    assert ticket_gate.tickets == {}
    assert not ticket_gate.fallbacks


@pytest.mark.parametrize("late", [0.0, 0.99, 1.0, 1.01, 30.0, 3600.0])
def test_completed_renewal_polled_late_is_ready_not_failed(gate, late):
    """The Store answered in 5 ms; the scheduler only came back `late` s later."""
    ticket_gate, executor, clock = gate
    owner = object()
    assert ticket_gate.request(owner, "a", ("key",), (0, 256)) is RenewalState.PENDING
    clock.now += 0.005
    executor.finish(0, 3)
    clock.now += late
    assert ticket_gate.request(owner, "a", ("key",), (0, 256)) is RenewalState.READY
    assert not ticket_gate.fallbacks  # an answered renewal is not a fallback
    assert not ticket_gate.tickets


@pytest.mark.parametrize("lateness", [0.05, 1.0, 5.0])
def test_burst_of_renewals_on_one_serial_thread_all_dispatch(gate, lateness):
    """60 loads x 30 ms on the single renewal thread, a step every 50 ms."""
    ticket_gate, executor, clock = gate
    ticket_gate.capacity = 60
    owners = [object() for _ in range(60)]
    for i, owner in enumerate(owners):
        clock.now = i * 0.05
        assert (
            ticket_gate.request(owner, f"s{i}", (f"k{i}",), ()) is RenewalState.PENDING
        )
    for i in range(60):  # the serial thread has answered every one
        executor.finish(i)
    clock.now += lateness
    states = {
        ticket_gate.request(owners[i], f"s{i}", (f"k{i}",), ()) for i in range(60)
    }
    assert states == {RenewalState.READY}
    assert not ticket_gate.fallbacks
    assert not ticket_gate.tickets


@pytest.mark.parametrize("cause", ["rpc_error", "cancelled"])
def test_failed_renewal_falls_back_to_dispatch_and_is_counted(gate, caplog, cause):
    ticket_gate, executor, _ = gate
    owner = object()
    assert ticket_gate.request(owner, "a", ("key",), (0, 256)) is RenewalState.PENDING
    future, _, _ = executor.jobs[0]
    if cause == "rpc_error":
        future.set_exception(RuntimeError("native RPC failed"))
    else:
        future.cancel()
    with caplog.at_level("WARNING", logger="atom"):
        state = ticket_gate.request(owner, "a", ("key",), (0, 256))
    assert state is RenewalState.READY  # the sync path dispatches the GET too
    assert ticket_gate.fallbacks == {
        ("error" if cause == "rpc_error" else "cancelled"): 1
    }
    assert "dispatching the load anyway" in caplog.text
    assert not ticket_gate.tickets


def test_timeout_dispatches_and_a_hung_rpc_does_not_fail_later_loads(gate):
    ticket_gate, executor, clock = gate
    a, b, c = object(), object(), object()
    assert ticket_gate.request(a, "a", ("a",), ()) is RenewalState.PENDING
    future = executor.jobs[0][0]
    assert future.set_running_or_notify_cancel()  # the RPC hangs
    clock.now = 0.99
    assert ticket_gate.request(a, "a", ("a",), ()) is RenewalState.PENDING
    clock.now = 1.0
    assert ticket_gate.request(a, "a", ("a",), ()) is RenewalState.READY
    assert ticket_gate.fallbacks == {"timeout": 1}
    # Behind the hung RPC every later hit dispatches at once, like the sync
    # path that skips renewal while one is running; none waits or fails.
    for sid, owner in (("b", b), ("c", c)):
        assert ticket_gate.request(owner, sid, (sid,), ()) is RenewalState.READY
    assert len(executor.jobs) == 1
    assert ticket_gate.fallbacks == {"timeout": 1, "renewal_busy": 2}
    future.set_result(0)  # the Store recovers: renewals resume
    assert ticket_gate.request(b, "b", ("b",), ()) is RenewalState.PENDING
    assert len(executor.jobs) == 2


def test_abort_of_running_renewal_then_new_load_does_not_wait(gate):
    ticket_gate, executor, _ = gate
    a, b = object(), object()
    assert ticket_gate.request(a, "a", ("a",), ()) is RenewalState.PENDING
    assert executor.jobs[0][0].set_running_or_notify_cancel()  # RPC hangs
    ticket_gate.cancel("a", a)  # abort: a running future cannot be stopped
    assert not ticket_gate.tickets
    # The next load must not queue behind the hung RPC and wait a timeout.
    assert ticket_gate.request(b, "b", ("b",), ()) is RenewalState.READY
    assert ticket_gate.fallbacks == {"renewal_busy": 1}
    assert len(executor.jobs) == 1


def test_queued_renewal_that_times_out_is_cancelled_and_dispatches(gate):
    ticket_gate, executor, clock = gate
    owner = object()
    assert ticket_gate.request(owner, "a", ("a",), ()) is RenewalState.PENDING
    clock.now = 2.0
    assert ticket_gate.request(owner, "a", ("a",), ()) is RenewalState.READY
    assert executor.jobs[0][0].cancelled()  # it never runs, nor blocks the thread
    assert not ticket_gate.stuck


@pytest.mark.parametrize("change", ["owner", "keys", "physical_blocks"])
def test_lifecycle_or_physical_identity_change_replaces_the_ticket(gate, change):
    ticket_gate, executor, _ = gate
    owner, keys, blocks = object(), ("key",), (0, 256, (4, 5))
    assert ticket_gate.request(owner, "a", keys, blocks) is RenewalState.PENDING
    old = executor.jobs[0][0]
    if change == "owner":
        owner = object()
    elif change == "keys":
        keys = ("other",)
    else:
        blocks = (0, 256, (6, 7))
    assert ticket_gate.request(owner, "a", keys, blocks) is RenewalState.PENDING
    assert old.cancelled()
    assert len(executor.jobs) == 2


def test_cancellation_close_and_capacity_are_bounded(gate):
    ticket_gate, executor, _ = gate
    a, b = object(), object()
    assert ticket_gate.request(a, "a", ("a",), ()) is RenewalState.PENDING
    assert ticket_gate.request(b, "b", ("b",), ()) is RenewalState.PENDING
    # A full renewal queue dispatches like the sync path instead of failing.
    assert ticket_gate.request(object(), "c", ("c",), ()) is RenewalState.READY
    assert ticket_gate.fallbacks == {"capacity": 1}
    assert len(executor.jobs) == 2
    ticket_gate.cancel("a", object())  # An old owner cannot cancel a reused ID.
    assert "a" in ticket_gate.tickets
    ticket_gate.cancel("a", a)
    assert executor.jobs[0][0].cancelled()
    ticket_gate.close()
    assert executor.jobs[1][0].cancelled()
    assert not ticket_gate.tickets
    assert ticket_gate.request(a, "a", ("a",), ()) is RenewalState.READY
    assert ticket_gate.fallbacks["closed"] == 1
    assert len(executor.jobs) == 2


def test_rejected_submission_dispatches(gate):
    ticket_gate, executor, _ = gate
    executor.submit = lambda fn, keys: None
    assert ticket_gate.request(object(), "a", ("a",), ()) is RenewalState.READY
    assert ticket_gate.fallbacks == {"rejected": 1}
    assert not ticket_gate.tickets


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("off", False),
        ("1", True),
        ("true", True),
        (" TRUE ", True),
    ],
)
def test_async_renew_flag_is_opt_in(monkeypatch, raw, expected):
    from atom.utils import envs

    if raw is None:
        monkeypatch.delenv("ATOM_STORE_ASYNC_RENEW", raising=False)
    else:
        monkeypatch.setenv("ATOM_STORE_ASYNC_RENEW", raw)
    assert envs.ATOM_STORE_ASYNC_RENEW is expected
