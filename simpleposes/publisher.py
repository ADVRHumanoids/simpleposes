"""Publishing link poses: one PUB socket, one message per publish() with the poses of all links."""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Callable

import numpy as np
import zmq

from .wire import encode_names, encode_snapshot


class PosePublisher:
    """Publishes the poses of a fixed set of links, each in its own reference frame.

    Every message carries the link and frame names, so subscribers need no setup and survive publisher restarts.
    A socket is not thread-safe: call publish() from one thread at a time.
    """

    def __init__(self, links : Mapping[str, str], endpoint : str, clock : Callable[[], float] = time.time,
                 context : zmq.Context | None = None):
        """links maps each link to the reference frame its pose is expressed in. endpoint is the address the PUB
        socket binds to, e.g. "tcp://*:5800" (a "*" port picks a free one, see self.endpoint). clock stamps the send
        time of each message."""
        if not links:
            raise ValueError("no links to publish")
        self.links : list[str] = list(links)
        self.frames : list[str] = [links[link] for link in self.links]
        self._names = encode_names(self.links, self.frames)
        self._clock = clock
        self._seq = 0
        self._ctx = context or zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)
        self.endpoint : str = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
        """Actual endpoint of the socket (with the port resolved if it was "*")."""

    def publish(self, poses, t : float) -> None:
        """Publish the poses of all links, stamped with t (simulation or real time).

        poses is either an (n_links, 7) array in the order of self.links, or a dict mapping every link to its pose.
        Each pose is x, y, z, qx, qy, qz, qw, in the link's reference frame."""
        if isinstance(poses, Mapping):
            if poses.keys() != set(self.links):
                missing = set(self.links) - poses.keys()
                unknown = poses.keys() - set(self.links)
                raise ValueError(f"poses must cover exactly the published links: "
                                 f"missing {sorted(missing)}, unknown {sorted(unknown)}")
            poses = np.stack([np.asarray(poses[link], dtype=np.float64) for link in self.links])
        else:
            poses = np.asarray(poses, dtype=np.float64)
        if poses.shape != (len(self.links), 7):
            raise ValueError(f"poses must be ({len(self.links)}, 7), got {poses.shape}")
        self._socket.send(encode_snapshot(self._names, self._seq, t, self._clock(), poses), copy=False)
        self._seq += 1

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> PosePublisher:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
