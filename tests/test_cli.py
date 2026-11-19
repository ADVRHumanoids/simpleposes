"""simpleposes-echo against a publisher on local TCP."""
import threading
import time

import numpy as np
import pytest

from simpleposes import PosePublisher
from simpleposes.cli import main

POSE = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]


@pytest.fixture
def publisher():
    pub = PosePublisher({"base" : "world", "ee" : "base"}, "tcp://127.0.0.1:*")
    stop = threading.Event()

    def run(): # the socket is only used by this thread until it stops
        t = 0.0
        while not stop.is_set():
            pub.publish(np.tile(POSE, (2, 1)), t=t)
            t += 0.01
            time.sleep(0.005)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    yield pub
    stop.set()
    thread.join(timeout=2.0)
    pub.close()


def test_prints_the_latest_snapshot(publisher, capsys):
    assert main(["--connect", publisher.endpoint, "--rate", "20", "--count", "3"]) == 0
    out = capsys.readouterr().out
    headers = [line for line in out.splitlines() if line.startswith("t ")]
    assert len(headers) == 3
    assert "rate" in headers[-1] and "age" in headers[-1]
    assert "  base  in world  xyz +1.0000 +2.0000 +3.0000  qxyzw +0.0000 +0.0000 +0.0000 +1.0000" in out
    assert "  ee    in base  xyz" in out


def test_prints_only_the_requested_links(publisher, capsys):
    main(["--connect", publisher.endpoint, "--rate", "20", "--links", "ee", "--count", "1"])
    out = capsys.readouterr().out
    assert "ee  in base" in out and "in world" not in out


def test_rejects_links_that_are_not_published(publisher):
    with pytest.raises(SystemExit, match=r"\['nope'\] are not published"):
        main(["--connect", publisher.endpoint, "--rate", "20", "--links", "nope", "--count", "1"])


def test_all_prints_every_snapshot_as_numbers(publisher, capsys):
    main(["--connect", publisher.endpoint, "--all", "--count", "5"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "# t seq " + " ".join(f"{link}.{c}" for link in ("base", "ee")
                                             for c in ("x", "y", "z", "qx", "qy", "qz", "qw"))
    assert lines[1] == "# frames: base:world ee:base"
    rows = np.array([[float(v) for v in line.split()] for line in lines[2:]])
    assert rows.shape == (5, 16)
    assert np.all(np.diff(rows[:, 1]) == 1)
    np.testing.assert_array_equal(rows[:, 2:], np.tile(POSE*2, (5, 1)))


@pytest.mark.parametrize("options", [["--rate", "0"], ["--count", "0"], []])
def test_rejects_bad_options(options):
    with pytest.raises(SystemExit) as exc:
        main(options if options else ["--rate", "1"]) # [] -> missing --connect
    assert exc.value.code == 2
