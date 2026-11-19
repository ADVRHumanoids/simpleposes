import math
import struct

import numpy as np
import pytest

from simpleposes.wire import VERSION, decode_snapshot, encode_names, encode_snapshot

LINKS = ["base", "ee"]
FRAMES = ["world", "base"]


def _poses(n):
    return np.arange(n*7, dtype=np.float64).reshape(n, 7)


def _message(links=LINKS, frames=FRAMES):
    return encode_snapshot(encode_names(links, frames), seq=3, t=1.5, t_send=100.25, poses=_poses(len(links)))


def test_roundtrip():
    s = decode_snapshot(_message())
    assert (s.seq, s.t, s.t_send) == (3, 1.5, 100.25)
    assert s.links == LINKS and s.frames == FRAMES
    np.testing.assert_array_equal(s.poses, _poses(2))
    assert s.frame("ee") == "base"
    np.testing.assert_array_equal(s.position("ee"), [7, 8, 9])
    np.testing.assert_array_equal(s.orientation("ee"), [10, 11, 12, 13])
    assert set(s.as_dict()) == set(LINKS) and s.as_dict()["ee"][0] == "base"
    assert math.isnan(s.t_recv)


def test_poses_are_read_in_place():
    s = decode_snapshot(_message(["a"], ["w"])) # odd names length: the poses need padding
    assert not s.poses.flags.writeable
    np.testing.assert_array_equal(s.poses, _poses(1))


def test_unicode_names():
    s = decode_snapshot(_message(["pinza_ü", "ee"], ["mondo", "pinza_ü"]))
    assert s.links == ["pinza_ü", "ee"] and s.frame("ee") == "pinza_ü"


@pytest.mark.parametrize("links", [["a\0b"], [""]])
def test_rejects_bad_names(links):
    with pytest.raises(ValueError, match="names"):
        encode_names(links, ["w"])


def test_rejects_wrong_pose_shape():
    with pytest.raises(ValueError, match="n_links, 7"):
        encode_snapshot(encode_names(LINKS, FRAMES), 0, 0.0, 0.0, np.zeros((2, 6)))


def test_rejects_foreign_message():
    with pytest.raises(ValueError, match="not a simpleposes"):
        decode_snapshot(b"XXXX" + bytes(60))


def test_rejects_other_version():
    msg = bytearray(_message())
    struct.pack_into("<H", msg, 4, VERSION + 1)
    with pytest.raises(ValueError, match="wire version"):
        decode_snapshot(msg)


@pytest.mark.parametrize("cut", [8, 1])
def test_rejects_truncated(cut):
    with pytest.raises(ValueError, match="bytes"):
        decode_snapshot(_message()[:-cut])


def test_rejects_short():
    with pytest.raises(ValueError, match="too short"):
        decode_snapshot(b"SPOS")
