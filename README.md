# simpleposes

Link poses, each expressed in its own reference frame, from a publisher to any number of subscribers.

- The **publisher** is created with a fixed set of links, each with its reference frame. Every `publish()` sends the
  poses of all links as one snapshot, stamped with the time given to it (simulation or real time).
- The **subscriber** gets the latest snapshot (or every snapshot in order, for logging) as plain numpy arrays.
- Each message carries the link and frame names, so subscribers need no setup and survive publisher restarts.

It only needs `pyzmq` and `numpy`. It does not compose frames or interpolate in time: each pose is in the reference
frame the publisher declared for it.

## Install

```bash
uv pip install -e .
```

## Use from Python

Poses are `x, y, z, qx, qy, qz, qw` (quaternion in xyzw order).

```python
from simpleposes import PosePublisher, PoseSubscriber

pub = PosePublisher({"base_link" : "world", "ee_link" : "base_link"}, endpoint="tcp://*:5800")
pub.publish(poses, t=sim_time)                   # (2, 7) array, in the order of the links above
pub.publish({"base_link" : p_base, "ee_link" : p_ee}, t=sim_time)  # or a dict covering every link

sub = PoseSubscriber("tcp://robot-pc:5800")      # conflate=False to receive every snapshot, in order
snapshot = sub.latest(timeout_s=1.0)             # newest snapshot so far, None if nothing arrived yet
snapshot = sub.recv(timeout_s=1.0)               # or wait for the next one
snapshot.t                                       # the stamp given to publish()
snapshot.pose("ee_link")                         # (7,) array
snapshot.frame("ee_link")                        # "base_link"
snapshot.position("ee_link"), snapshot.orientation("ee_link")
snapshot.as_dict()                               # {link : (frame, pose)}
```

`snapshot.t_send` is when the message was sent and `snapshot.t_recv` when it arrived, both `time.time()` by default
(`clock=` on either side to change it). Their difference is the latency only if the two machines' clocks are synced.

As with any ZMQ subscriber, snapshots published before the subscriber has connected are lost. A publisher is not
thread-safe: call `publish()` from one thread at a time.

## Command line

`simpleposes-echo` prints what a publisher publishes:

```
simpleposes-echo --connect tcp://robot-pc:5800                  # latest snapshot, twice a second
simpleposes-echo --connect tcp://robot-pc:5800 --links ee_link --rate 10
simpleposes-echo --connect tcp://robot-pc:5800 --all > poses.txt  # every snapshot, one line each
```

The default output shows the stamp, the seq number, the publishing rate (from the seq numbers, so conflation does
not lower it), the age (arrival minus send time, meaningful only with synced clocks), and each link's reference
frame, position and quaternion (xyzw). With `--all` each line is `t seq` followed by `x y z qx qy qz qw` of each
link, after two `#` header lines naming the columns and the reference frames. `--count N` exits after N snapshots.

## Wire format

One flat message per snapshot (see `simpleposes/wire.py`): a fixed header (magic, version, seq, t, t_send, number of
links), the link and frame names, and an `(n_links, 7)` float64 block. Publisher and subscriber must have the same
wire version; a mismatch is reported on receive.

## Tests

```bash
python -m pytest tests
```
