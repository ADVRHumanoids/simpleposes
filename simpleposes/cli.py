"""simpleposes-echo: print the link poses a PosePublisher publishes."""
from __future__ import annotations

import argparse
import sys
import time
from typing import TextIO

import numpy as np

from .subscriber import PoseSubscriber
from .wire import PoseSnapshot


def _positive_float(value : str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _positive_int(value : str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parse_args(argv : list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="simpleposes-echo",
                                description="Print the link poses a simpleposes publisher publishes.")
    p.add_argument("--connect", required=True, metavar="ENDPOINT",
                   help="publisher address, e.g. tcp://robot-pc:5800")
    p.add_argument("--links", nargs="+", metavar="LINK", help="print only these links (default: all)")
    p.add_argument("--rate", type=_positive_float, default=2.0, metavar="HZ",
                   help="prints per second, of the latest snapshot (default: 2)")
    p.add_argument("--all", action="store_true",
                   help="print every snapshot, one line each: t, seq, then x y z qx qy qz qw of each link")
    p.add_argument("--count", type=_positive_int, metavar="N", help="exit after printing N snapshots")
    return p.parse_args(argv)


def _select_links(snapshot : PoseSnapshot, wanted : list[str] | None) -> list[str]:
    if wanted is None:
        return list(snapshot.links)
    missing = [link for link in wanted if link not in snapshot.links]
    if missing:
        raise SystemExit(f"links {missing} are not published; published: {snapshot.links}")
    return list(wanted)


def _numbers(values) -> str:
    return " ".join(f"{v:+.4f}" for v in values)


def _format(snapshot : PoseSnapshot, links : list[str], rate_hz : float | None) -> str:
    rate = "" if rate_hz is None else f"  rate {rate_hz:.1f} Hz"
    lines = [f"t {snapshot.t:.4f}  seq {snapshot.seq}{rate}  age {1000*(snapshot.t_recv - snapshot.t_send):.1f} ms"]
    width = max(len(link) for link in links)
    for link in links:
        pose = snapshot.pose(link)
        lines.append(f"  {link:<{width}}  in {snapshot.frame(link)}  xyz {_numbers(pose[:3])}  qxyzw {_numbers(pose[3:])}")
    return "\n".join(lines)


def _echo_latest(sub : PoseSubscriber, args : argparse.Namespace, out : TextIO) -> None:
    """Print the latest snapshot args.rate times per second. The rate shown is that of publishing, from the seq
    numbers, so it is right also when snapshots are dropped by conflation."""
    period = 1.0/args.rate
    printed = 0
    last : PoseSnapshot | None = None
    last_new = next_print = time.monotonic()
    while args.count is None or printed < args.count:
        time.sleep(max(0.0, next_print - time.monotonic()))
        next_print += period
        snapshot = sub.latest()
        now = time.monotonic()
        if snapshot is None or (last is not None and snapshot.seq == last.seq):
            print(f"no new snapshot from {sub.endpoint} for {now - last_new:.1f} s", file=out, flush=True)
            continue
        rate = None
        if last is not None and snapshot.seq > last.seq and snapshot.t_recv > last.t_recv:
            rate = (snapshot.seq - last.seq)/(snapshot.t_recv - last.t_recv)
        print(_format(snapshot, _select_links(snapshot, args.links), rate), file=out, flush=True)
        last = snapshot
        last_new = now
        printed += 1


def _echo_all(sub : PoseSubscriber, args : argparse.Namespace, out : TextIO) -> None:
    """Print every snapshot as one line of numbers, after a two-line header (columns, reference frames)."""
    printed = 0
    links = None
    while args.count is None or printed < args.count:
        snapshot = sub.recv(timeout_s=1.0)
        if snapshot is None:
            continue
        if links is None:
            links = _select_links(snapshot, args.links)
            print("# t seq " + " ".join(f"{link}.{c}" for link in links for c in ("x", "y", "z", "qx", "qy", "qz", "qw")),
                  file=out)
            print("# frames: " + " ".join(f"{link}:{snapshot.frame(link)}" for link in links), file=out)
        row = np.concatenate([snapshot.pose(link) for link in links])
        print(f"{snapshot.t!r} {snapshot.seq} " + " ".join(repr(float(v)) for v in row), file=out, flush=True)
        printed += 1


def main(argv : list[str] | None = None) -> int:
    args = _parse_args(argv)
    with PoseSubscriber(args.connect, conflate=not args.all) as sub:
        try:
            (_echo_all if args.all else _echo_latest)(sub, args, sys.stdout)
        except KeyboardInterrupt:
            pass
    return 0
