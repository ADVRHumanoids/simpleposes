"""Wire format of simpleposes messages.

A snapshot of all published links travels as ONE flat message, because ZMQ_CONFLATE (which gives a subscriber the
latest snapshot instead of a backlog) does not support multipart messages. Layout:

    header  struct "<4sHHQddII": magic, version, reserved, seq, t, t_send, n_links, names_len
    names   utf-8, names_len bytes: each link followed by its reference frame, NUL-separated
            (link_0, frame_0, link_1, frame_1, ...)
    poses   float64 (n_links, 7): x, y, z, qx, qy, qz, qw of each link in its reference frame

poses starts at an offset aligned to 8 bytes, so decoding reads it in place with np.frombuffer, without copies.
t is the stamp given to the publisher (simulation or real time); t_send is when the message was sent, in the
publisher's clock.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Sequence

import numpy as np

MAGIC = b"SPOS"
VERSION = 1
_HEADER = struct.Struct("<4sHHQddII")
_ALIGN = 8
_POSE_DTYPE = np.dtype("<f8")


def _pad(offset : int) -> int:
    return (-offset) % _ALIGN


@dataclass
class PoseSnapshot:
    """The poses of all published links at one instant."""
    seq : int
    t : float
    """The stamp given to PosePublisher.publish (simulation or real time)."""
    t_send : float
    """When the message was sent, in the publisher's clock."""
    links : list[str]
    frames : list[str]
    """Reference frame of each link, in the order of links."""
    poses : np.ndarray
    """(n_links, 7) float64: x, y, z, qx, qy, qz, qw of each link in its reference frame. Read-only when received."""
    t_recv : float = math.nan
    """When it arrived, in the subscriber's clock. Not on the wire: NaN until a PoseSubscriber receives it."""

    def __post_init__(self):
        self._index = {link : i for i, link in enumerate(self.links)}

    def pose(self, link : str) -> np.ndarray:
        """(7,) x, y, z, qx, qy, qz, qw of link, in frame(link)."""
        return self.poses[self._index[link]]

    def position(self, link : str) -> np.ndarray:
        return self.pose(link)[:3]

    def orientation(self, link : str) -> np.ndarray:
        """Quaternion, xyzw order."""
        return self.pose(link)[3:]

    def frame(self, link : str) -> str:
        return self.frames[self._index[link]]

    def as_dict(self) -> dict[str, tuple[str, np.ndarray]]:
        """{link : (reference frame, (7,) pose)}"""
        return {link : (self.frames[i], self.poses[i]) for i, link in enumerate(self.links)}


def encode_names(links : Sequence[str], frames : Sequence[str]) -> bytes:
    """The names block of a message. It depends only on the published links, so a publisher encodes it once."""
    if len(links) != len(frames):
        raise ValueError(f"{len(links)} links but {len(frames)} frames")
    for name in (*links, *frames):
        if not name or "\0" in name:
            raise ValueError(f"link and frame names must be non-empty and contain no NUL, got {name!r}")
    return "\0".join(name for pair in zip(links, frames) for name in pair).encode("utf-8")


def encode_snapshot(names : bytes, seq : int, t : float, t_send : float, poses : np.ndarray) -> bytes:
    """Pack a snapshot into a single message. names comes from encode_names, poses is (n_links, 7)."""
    poses = np.ascontiguousarray(poses, dtype=_POSE_DTYPE)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"poses must be (n_links, 7), got {poses.shape}")
    offset = _HEADER.size + len(names)
    return b"".join([_HEADER.pack(MAGIC, VERSION, 0, seq, float(t), float(t_send), poses.shape[0], len(names)),
                     names,
                     b"\0"*_pad(offset),
                     memoryview(poses).cast("B")])


def decode_snapshot(buf) -> PoseSnapshot:
    """Unpack a message. poses is a read-only view into buf (e.g. a zmq.Frame's buffer, received with copy=False),
    which it keeps alive."""
    mv = memoryview(buf).cast("B")
    if len(mv) < _HEADER.size:
        raise ValueError(f"message too short for a simpleposes header ({len(mv)} bytes)")
    magic, version = struct.unpack_from("<4sH", mv, 0)
    if magic != MAGIC:
        raise ValueError(f"not a simpleposes message (magic {magic!r})")
    if version != VERSION: # checked before unpacking the rest: the header layout may change between versions
        raise ValueError(f"simpleposes wire version {version}, this receiver understands {VERSION}: "
                         f"update simpleposes on both sides")
    _, _, _, seq, t, t_send, n_links, names_len = _HEADER.unpack_from(mv, 0)
    offset = _HEADER.size + names_len
    offset += _pad(offset)
    expected = offset + n_links*7*_POSE_DTYPE.itemsize
    if expected != len(mv):
        raise ValueError(f"simpleposes message is {len(mv)} bytes, the header describes {expected}")
    names = bytes(mv[_HEADER.size:_HEADER.size+names_len]).decode("utf-8").split("\0") if names_len else []
    if len(names) != 2*n_links:
        raise ValueError(f"simpleposes message has {n_links} links but {len(names)} names")
    poses = np.frombuffer(mv, dtype=_POSE_DTYPE, count=n_links*7, offset=offset).reshape(n_links, 7)
    return PoseSnapshot(seq=seq, t=t, t_send=t_send, links=names[0::2], frames=names[1::2], poses=poses)
