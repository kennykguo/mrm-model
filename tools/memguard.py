"""Run a command under a memory watchdog and a machine-wide lock for heavy solver jobs.

Usage:  .venv/bin/python tools/memguard.py [--limit-gb N] [--heavy] -- <command> [args...]

Read this first. The machine has 16 GB of RAM and several solver jobs may run at once. Two guards protect it:

- The memory watchdog. Every 0.5 s it adds up the resident memory (RSS) of the command and all its child processes.
  When the total is above the limit, it kills the whole process group and exits with code 137.
- The heavy lock. With --heavy, the job first takes an exclusive lock (a directory under /tmp), so that only one
  heavy solver job runs at a time. Light jobs skip the lock. A lock left behind by a dead process is removed. The
  lock directory appears together with its owner file (take_heavy_lock), so a waiter never mistakes a lock that is
  being taken for a stale one.

Default limit: 6 GB for --heavy jobs (the lock makes sure that only one runs at a time) and 3 GB otherwise. An
explicit --limit-gb always wins. The exit code is the exit code of the command.
"""
import argparse, errno, os, shutil, signal, subprocess, sys, time

LOCK = "/tmp/mrm-model-heavy.lock"     # lock directory; its file "owner" holds "<pid> <command>" of the holder
POLL_S = 0.5                           # seconds between two memory checks
LOCK_RETRY_S = 5                       # seconds between two attempts to take the heavy lock


def tree_rss_bytes(root_pid):
    """Total resident memory (bytes) of process ``root_pid`` and all its descendants, from one ``ps`` call."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    children, rss = {}, {}
    for line in out.split("\n"):
        cols = line.split()
        if len(cols) == 3:
            pid, ppid, rss_kib = map(int, cols)
            children.setdefault(ppid, []).append(pid)
            rss[pid] = rss_kib * 1024
    total, stack = 0, [root_pid]
    while stack:
        pid = stack.pop()
        total += rss.get(pid, 0)
        stack.extend(children.get(pid, []))
    return total


def take_heavy_lock(cmd, lock=LOCK):
    """Wait until this process holds the heavy lock: the directory ``lock`` whose file "owner" holds
    "<pid> <command>".

    The owner file is written first into a private directory ``<lock>.<pid>``, which is then renamed to ``lock``.
    The rename is atomic and fails when ``lock`` exists with an owner file in it, so the lock never exists without
    its owner. A waiter removes the lock only when its owner process is dead or the owner file holds no pid. When
    the owner file is not there (the holder is removing the lock right now), the waiter waits and tries again.
    """
    private_dir = f"{lock}.{os.getpid()}"
    waited = 0
    try:
        while True:
            os.makedirs(private_dir, exist_ok=True)
            with open(os.path.join(private_dir, "owner"), "w") as f:
                f.write(f"{os.getpid()} {' '.join(cmd)}\n")
            try:
                os.rename(private_dir, lock)
                return
            except OSError as e:
                if e.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
            try:
                owner = int(open(os.path.join(lock, "owner")).read().split()[0])
                os.kill(owner, 0)            # signal 0 only checks that the owner is still alive
            except FileNotFoundError:
                pass                         # the lock is being released: wait, then try again
            except (OSError, ValueError, IndexError):
                subprocess.run(["rm", "-rf", lock])     # stale lock: the owner is dead or the file holds no pid
                continue
            if waited % 60 == 0:
                print(f"[memguard] waiting for heavy lock ({waited}s)", file=sys.stderr, flush=True)
            time.sleep(LOCK_RETRY_S); waited += LOCK_RETRY_S
    finally:
        shutil.rmtree(private_dir, ignore_errors=True)  # left only when the rename did not happen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-gb", type=float, default=None)
    ap.add_argument("--heavy", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if args.limit_gb is None:
        args.limit_gb = 6.0 if args.heavy else 3.0
    if args.heavy:
        take_heavy_lock(cmd)
    try:
        # A new session makes the command the leader of its own process group, so killpg reaches every child.
        proc = subprocess.Popen(cmd, start_new_session=True)
        peak = 0
        while proc.poll() is None:
            rss = tree_rss_bytes(proc.pid)
            peak = max(peak, rss)
            if rss > args.limit_gb * 1e9:
                print(f"[memguard] killed: RSS {rss/1e9:.2f} GB > {args.limit_gb} GB", file=sys.stderr, flush=True)
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                sys.exit(137)
            time.sleep(POLL_S)
        print(f"[memguard] exit {proc.returncode}, peak RSS {peak/1e9:.2f} GB", file=sys.stderr, flush=True)
        sys.exit(proc.returncode)
    finally:
        if args.heavy:
            subprocess.run(["rm", "-rf", LOCK])


if __name__ == "__main__":
    main()
