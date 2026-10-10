"""mrm.sim: the engine (kernel, runner, metrics), experiments and run manifests, the gym environment, the binary
bridge and golden vectors (docs/ENGINE.md). Submodules are imported on use. ``mrm.sim.record_vectors`` (07 P1-21)
is a lazy export of :func:`mrm.sim.vectors.record_vectors`, so ``import mrm.sim`` stays light."""

__all__ = ["record_vectors"]


def __getattr__(name):
    if name == "record_vectors":
        from mrm.sim.vectors import record_vectors
        return record_vectors
    raise AttributeError(f"module 'mrm.sim' has no attribute {name!r}")
