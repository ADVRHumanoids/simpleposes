"""Receiving link poses: the latest snapshot, or every snapshot in order."""
from __future__ import annotations

import time
from typing import Callable

import zmq

from .wire import PoseSnapshot, decode_snapshot


class PoseSubscriber:
    """Receives the snapshots of a PosePublisher.

    With conflate=True (the default) only the latest snapshot is kept, for controllers and viewers. With
    conflate=False every snapshot is queued (up to the socket's high-water mark), for logging.
    As with any ZMQ subscriber, snapshots published before the connection is established are lost.
    """

    def __init__(self, endpoint : str, conflate : bool = True, clock : Callable[[], float] = time.time,
                 context : zmq.Context | None = None):
        """endpoint is the publisher's address, e.g. "tcp://robot-pc:5800". clock stamps the arrival time of each
        snapshot (PoseSnapshot.t_recv)."""
        self.endpoint = endpoint
        self._clock = clock
        self._latest : PoseSnapshot | None = None
        self._ctx = context or zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        if conflate:
            self._socket.setsockopt(zmq.CONFLATE, 1) # must be set before connecting
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(endpoint)

    def _receive(self) -> PoseSnapshot:
        snapshot = decode_snapshot(self._socket.recv(copy=False).buffer)
        snapshot.t_recv = self._clock()
        self._latest = snapshot
        return snapshot

    def recv(self, timeout_s : float | None = None) -> PoseSnapshot | None:
        """Wait for the next snapshot, at most timeout_s (None: forever). None on timeout."""
        timeout_ms = None if timeout_s is None else max(0, int(timeout_s*1000))
        if not self._socket.poll(timeout_ms):
            return None
        return self._receive()

    def latest(self, timeout_s : float = 0.0) -> PoseSnapshot | None:
        """The newest snapshot received so far, after taking in all that have arrived. If none was ever received,
        wait for one at most timeout_s; None if still nothing."""
        while self._socket.poll(0):
            self._receive()
        if self._latest is None and timeout_s > 0:
            self.recv(timeout_s)
        return self._latest

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> PoseSubscriber:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
