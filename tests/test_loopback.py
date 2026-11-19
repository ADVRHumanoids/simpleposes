"""Publisher -> subscriber over local TCP."""
import time

import numpy as np
import pytest

from simpleposes import PosePublisher, PoseSubscriber

LINKS = {"base" : "world", "ee" : "base"}


def _poses():
    return np.arange(14, dtype=np.float64).reshape(2, 7)


def _connect(pub, sub):
    """PUB drops messages until the subscription has reached it: publish until one gets through, then drain."""
    deadline = time.monotonic() + 5.0
    while sub.recv(timeout_s=0.05) is None:
        if time.monotonic() > deadline:
            raise TimeoutError("subscriber did not connect")
        pub.publish(np.zeros((2, 7)), t=-1.0)
    while sub.recv(timeout_s=0.1) is not None:
        pass


@pytest.fixture
def pubsub():
    opened = []

    def make(conflate=True):
        pub = PosePublisher(LINKS, "tcp://127.0.0.1:*")
        sub = PoseSubscriber(pub.endpoint, conflate=conflate)
        opened.append((pub, sub))
        _connect(pub, sub)
        return pub, sub

    yield make
    for pub, sub in opened:
        sub.close()
        pub.close()


def test_array_arrives(pubsub):
    pub, sub = pubsub()
    pub.publish(_poses(), t=2.0)
    s = sub.recv(timeout_s=5.0)
    assert s is not None
    assert s.t == 2.0 and s.links == ["base", "ee"] and s.frames == ["world", "base"]
    np.testing.assert_array_equal(s.poses, _poses())
    assert s.t_recv >= s.t_send


def test_dict_arrives_in_link_order(pubsub):
    pub, sub = pubsub()
    poses = _poses()
    pub.publish({"ee" : poses[1], "base" : poses[0]}, t=3.0)
    s = sub.recv(timeout_s=5.0)
    np.testing.assert_array_equal(s.poses, poses)
    np.testing.assert_array_equal(s.pose("ee"), poses[1])


@pytest.mark.parametrize("poses, match", [({"base" : np.zeros(7)}, "missing \\['ee'\\]"),
                                          ({"base" : np.zeros(7), "ee" : np.zeros(7), "x" : np.zeros(7)},
                                           "unknown \\['x'\\]"),
                                          (np.zeros((2, 6)), "\\(2, 7\\)"),
                                          (np.zeros((3, 7)), "\\(2, 7\\)"),
                                          ({"base" : np.zeros(6), "ee" : np.zeros(6)}, "\\(2, 7\\)")])
def test_rejects_bad_poses(poses, match):
    with PosePublisher(LINKS, "tcp://127.0.0.1:*") as pub:
        with pytest.raises(ValueError, match=match):
            pub.publish(poses, t=0.0)


def test_conflate_keeps_only_the_latest(pubsub):
    pub, sub = pubsub(conflate=True)
    for i in range(20):
        pub.publish(_poses(), t=float(i))
    time.sleep(0.2)
    assert sub.latest().t == 19.0
    assert sub.recv(timeout_s=0.1) is None


def test_no_conflate_keeps_every_snapshot_in_order(pubsub):
    pub, sub = pubsub(conflate=False)
    for i in range(20):
        pub.publish(_poses(), t=float(i))
    received = [sub.recv(timeout_s=2.0) for _ in range(20)]
    assert [s.t for s in received] == [float(i) for i in range(20)]
    assert np.all(np.diff([s.seq for s in received]) == 1)


def test_latest_waits_for_the_first_snapshot():
    with PosePublisher(LINKS, "tcp://127.0.0.1:*") as pub, PoseSubscriber(pub.endpoint) as sub:
        assert sub.latest() is None
        deadline = time.monotonic() + 5.0
        snapshot = None
        while snapshot is None and time.monotonic() < deadline:
            pub.publish(_poses(), t=7.0)
            snapshot = sub.latest(timeout_s=0.05)
        assert snapshot is not None and snapshot.t == 7.0
