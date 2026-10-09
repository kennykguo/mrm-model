"""Run a command under a memory watchdog and a machine-wide lock for heavy solver jobs.

Usage:  .venv/bin/python tools/memguard.py [--limit-gb 5] [--heavy] -- <command> [args...]

The machine has 16 GB of RAM and several agents may run solvers at once. --heavy takes an exclusive
lock (a directory under /tmp) so that only one heavy solver job runs at a time; light jobs skip the
lock. Every job is killed (whole process group) if its total resident memory exceeds --limit-gb.
"""
import argparse, os, signal, subprocess, sys, time

LOCK = "/tmp/mrm-model-heavy.lock"

def tree_rss_bytes(root_pid):
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    kids, rss = {}, {}
    for line in out.split("\n"):
        p = line.split()
        if len(p) == 3:
            pid, ppid, r = map(int, p)
            kids.setdefault(ppid, []).append(pid)
            rss[pid] = r * 1024
    total, stack = 0, [root_pid]
    while stack:
        pid = stack.pop()
        total += rss.get(pid, 0)
        stack.extend(kids.get(pid, []))
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-gb", type=float, default=5.0)
    ap.add_argument("--heavy", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if a.heavy:
        waited = 0
        while True:
            try:
                os.mkdir(LOCK)
                with open(os.path.join(LOCK, "owner"), "w") as f:
                    f.write(f"{os.getpid()} {' '.join(cmd)}\n")
                break
            except FileExistsError:
                try:
                    owner = int(open(os.path.join(LOCK, "owner")).read().split()[0])
                    os.kill(owner, 0)
                except (OSError, ValueError, IndexError):
                    subprocess.run(["rm", "-rf", LOCK])
                    continue
                if waited % 60 == 0:
                    print(f"[memguard] waiting for heavy lock ({waited}s)", file=sys.stderr, flush=True)
                time.sleep(5); waited += 5
    try:
        p = subprocess.Popen(cmd, start_new_session=True)
        peak = 0
        while p.poll() is None:
            r = tree_rss_bytes(p.pid)
            peak = max(peak, r)
            if r > a.limit_gb * 1e9:
                print(f"[memguard] killed: RSS {r/1e9:.2f} GB > {a.limit_gb} GB", file=sys.stderr, flush=True)
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                sys.exit(137)
            time.sleep(0.5)
        print(f"[memguard] exit {p.returncode}, peak RSS {peak/1e9:.2f} GB", file=sys.stderr, flush=True)
        sys.exit(p.returncode)
    finally:
        if a.heavy:
            subprocess.run(["rm", "-rf", LOCK])

if __name__ == "__main__":
    main()
