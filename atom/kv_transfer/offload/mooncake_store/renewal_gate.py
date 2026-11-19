"""Scheduler-thread gate for native Store renewal.

Only immutable keys enter the existing renewal executor. Native read leases
expire by TTL; retiring a ticket is not a native unpin/remove operation.

The gate only decides *when* a load may be dispatched, never *whether*. The
synchronous path dispatches the GET whatever the renewal answered (the GET
validates what the Store holds), so every outcome here except "still running
and young" is ``READY``: a completed renewal (judged by ``done()`` first, not by
how late the scheduler polls), a timeout, an executor error, a full or stuck
renewal executor, or a closed gate. Those fallbacks are counted and logged.
"""

import logging
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from time import monotonic

logger = logging.getLogger("atom")


class RenewalState(Enum):
    PENDING = "pending"
    READY = "ready"


@dataclass
class _Ticket:
    owner: object
    identity: tuple
    future: object
    submitted: float


class RenewalGate:
    """No own thread. All ticket operations run on the scheduler thread."""

    def __init__(self, client, capacity=None, timeout_s=1.0, clock=monotonic):
        self.client = client
        self.capacity = client.capacity if capacity is None else capacity
        self.timeout_s = timeout_s
        self.clock = clock
        self.tickets = {}
        # Retired renewals whose RPC is still running: a hung Store call
        # must not make every later load wait out its own timeout.
        self.stuck = []
        self.fallbacks = Counter()
        self.closed = False

    def _call(self, keys):
        with self.client.backend_slot.take(renewal=True):
            return self.client.backend.renew_keys(keys)

    def _retire(self, sid):
        ticket = self.tickets.pop(sid, None)
        # A queued future is cancelled; a running one cannot be, so keep it:
        # its RPC still occupies the renewal worker, and later loads must not
        # each wait out a timeout behind it.
        if (
            ticket is not None
            and not ticket.future.cancel()
            and not ticket.future.done()
        ):
            self.stuck.append(ticket.future)
        # Running futures hold immutable keys only; no callbacks own Sequence.
        return ticket

    def cancel(self, sid, owner=None):
        ticket = self.tickets.get(sid)
        if ticket is not None and (owner is None or ticket.owner is owner):
            self._retire(sid)

    def _fallback(self, reason, sid, detail=None):
        """Dispatch the GET as the synchronous path does; count and log it."""
        self.fallbacks[reason] += 1
        n = self.fallbacks[reason]
        if n == 1 or n % 100 == 0:
            logger.warning(
                "Mooncake Store async renewal for %s: %s%s; dispatching the "
                "load anyway (%d so far)",
                sid,
                reason,
                f" ({detail!r})" if detail is not None else "",
                n,
            )
        return RenewalState.READY

    def request(self, owner, sid, keys, physical_identity):
        if self.closed:
            return self._fallback("closed", sid)
        identity = (tuple(keys), tuple(physical_identity))
        ticket = self.tickets.get(sid)
        if ticket is not None and (
            ticket.owner is not owner or ticket.identity != identity
        ):
            self._retire(sid)
            ticket = None
        if ticket is None:
            self.stuck = [f for f in self.stuck if not f.done()]
            if self.stuck:
                return self._fallback("renewal_busy", sid)
            if len(self.tickets) >= self.capacity:
                return self._fallback("capacity", sid)
            future = self.client.renew_executor.submit(self._call, identity[0])
            if future is None or future.cancelled():
                return self._fallback("rejected", sid)
            ticket = _Ticket(owner, identity, future, self.clock())
            self.tickets[sid] = ticket
        future = ticket.future
        # A finished renewal is judged by its outcome, never by how late the
        # scheduler polled it: the Store answered, whatever the wait.
        if future.done():
            self._retire(sid)
            if future.cancelled():
                return self._fallback("cancelled", sid)
            exc = future.exception()
            if exc is not None:
                return self._fallback("error", sid, exc)
            # Held == 0 also dispatches: the GET authoritatively reports it.
            return RenewalState.READY
        if self.clock() - ticket.submitted >= self.timeout_s:
            self._retire(sid)
            return self._fallback("timeout", sid)
        return RenewalState.PENDING

    def close(self):
        self.closed = True
        for sid in list(self.tickets):
            self._retire(sid)
