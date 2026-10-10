"""mrm: a calibrated, control-oriented plant model of Lightmatter-class silicon microring modulators."""
__version__ = "0.1.0"
__all__ = ["run", "__version__"]


def run(config, **kwargs):
    """Run an experiment (07 P1-17): ``mrm.run('experiments/examples/a1_pi_lock.yaml')`` or
    ``mrm.run(mrm.params.load('exp.yaml'))``.

    ``config``: an experiment YAML path or preset name, a loaded Config, or a mrm.sim.experiment.Experiment.
    Keyword arguments go to :func:`mrm.sim.experiment.run`: ``overrides`` ({path: value} or ["path=value"]),
    ``alts`` ({path: alternative}), ``options`` (experiment extension options, e.g. {"n_threads": 4}), ``kind``,
    ``runs_dir`` (default $MRM_RUNS_DIR or ./runs), ``write`` (False: no run directory), ``log``. Returns a
    mrm.sim.experiment.ExperimentResult (metrics, summary, report, run_dir). Every experiment key
    (experiment.start, record, acceptance, ...) is a mrm.params schema key, so both forms accept any experiment file
    (docs/handoff/experiments.md section 3).
    """
    from mrm.sim.experiment import run as _run
    return _run(config, **kwargs)
