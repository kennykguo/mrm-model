"""Smoke test: time-domain closed loop of 8 heater-tuned rings (numba), and the same plant behind a
gymnasium.Env (check_env + random rollout). Toy plant, numbers tagged:
  aggressor: on-die heater ramp 25 -> 105 C at 808 C/s, 100 ms period (G26, printed); the sawtooth shape
             with an instantaneous reset is ASSUMED;
  thermo-optic shift 86 pm/K (G26 ~15 GHz/C at 1310 nm, printed -> inferred in pm/K);
  FWHM 0.33 nm (G25 Q 3900-4200 read-off -> inferred);
  R_th 8.2 K/mW, tau 10 us (75 %) + 300 us (25 %), crosstalk 0.10 / 0.03 (all ASSUMED);
  PI at 1 MHz on the drop-port monitor photocurrent, heater-only actuation (P >= 0), 12-bit DAC.
Run:
  .venv/bin/python tools/memguard.py --limit-gb 3 -- .venv/bin/python archive/rl/research/t11_closed_loop_numba_gym.py
"""
import json, os, time
import numpy as np
import numba as nb
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

NR, DT = 8, 1e-6
TAU = np.array([10e-6, 300e-6]); SHARE = np.array([0.75, 0.25]); RTH = 8.2
X = np.eye(NR) + 0.10 * (np.eye(NR, k=1) + np.eye(NR, k=-1)) + 0.03 * (np.eye(NR, k=2) + np.eye(NR, k=-2))
DLDT, FWHM, PMAX = 0.086, 0.33, 40.0     # nm/K, nm, mW (heater full scale)
ALPHA = np.exp(-DT / TAU)


@nb.njit(cache=True)
def ambient(t):
    ph = (t % 0.1)
    return 25.0 + min(808.0 * ph, 80.0)


@nb.njit(cache=True)
def drop(det_nm):
    return 1.0 / (1.0 + (2.0 * det_nm / FWHM) ** 2)


@nb.njit(cache=True)
def simulate(nsteps, kp, ki, offs, xmat, alpha, share):
    """offs: cold detuning of each ring from its laser (nm, ring blue of laser). Returns RMS and max
    |detuning - target| over the last half (nm) and the final heater powers."""
    th = np.zeros((NR, 2))          # thermal states (K) per ring, two poles
    integ = np.zeros(NR)
    p = np.zeros(NR)
    target = -FWHM / 2.0 / np.sqrt(3.0)   # lock on the blue flank (max-slope point)
    dset = drop(target)
    acc2, mx, cnt = 0.0, 0.0, 0
    errs = np.zeros(nsteps * NR // 2 + NR)
    for n in range(nsteps):
        t = n * DT
        heat = xmat @ p                   # mW-equivalent seen by each ring after crosstalk
        for i in range(NR):
            for k in range(2):
                th[i, k] = alpha[k] * th[i, k] + (1 - alpha[k]) * share[k] * RTH * heat[i]
        amb = ambient(t) - 25.0
        for i in range(NR):
            det = offs[i] + DLDT * (th[i, 0] + th[i, 1] + amb)     # ring minus laser, nm
            meas = drop(det)                                        # normalised monitor photocurrent
            e = dset - meas
            if det > 0:                                             # red side: wrong flank, heat less
                e = -abs(e) - 0.5
            integ[i] = min(max(integ[i] + ki * e, 0.0), PMAX)       # clamped integrator (anti-windup)
            u = kp * e + integ[i]
            u = min(max(u, 0.0), PMAX)
            p[i] = np.round(u / PMAX * 4095.0) / 4095.0 * PMAX       # 12-bit DAC
            if n > nsteps // 2:
                d = det - target
                acc2 += d * d
                mx = max(mx, abs(d))
                errs[cnt] = abs(d)
                cnt += 1
    med = np.median(errs[:cnt])
    return np.sqrt(acc2 / cnt), mx, p, med


offs = -np.linspace(8.0, 8.4, NR)        # nm, cold rings sit blue of their lasers by more than the
                                          # 80 K x 86 pm/K = 6.9 nm aggressor excursion (ASSUMED)
args = (6.0, 0.6, offs, X, ALPHA, SHARE)


class RingBankEnv(gym.Env):
    """Observation: 8 normalised monitor photocurrents + ambient/100. Action: 8 heater powers in [0, 1]
    of full scale, held for 10 us (10 plant steps)."""
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = spaces.Box(0.0, 2.0, shape=(NR + 1,), dtype=np.float64)
        self.action_space = spaces.Box(0.0, 1.0, shape=(NR,), dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.th = np.zeros((NR, 2)); self.n = 0
        self.offs = offs + self.np_random.normal(0, 0.05, NR)
        return self._obs(), {}

    def _det(self):
        return self.offs + DLDT * (self.th.sum(1) + ambient(self.n * DT) - 25.0)

    def _obs(self):
        return np.concatenate([drop(self._det()), [ambient(self.n * DT) / 100.0]]).astype(np.float64)

    def step(self, action):
        heat = X @ (np.clip(action, 0, 1) * PMAX)
        for _ in range(10):
            self.th = ALPHA * self.th + (1 - ALPHA) * SHARE * RTH * heat[:, None]
            self.n += 1
        d = self._det() + FWHM / 2 / np.sqrt(3)
        reward = -float(np.mean(d**2))
        return self._obs(), reward, False, self.n >= 100_000, {}


def main():
    t0 = time.time(); simulate(1000, *args); compile_s = time.time() - t0
    t0 = time.time(); rms, mx, pfin, med = simulate(200_000, *args); run_s = time.time() - t0
    t0 = time.time(); simulate.py_func(2000, *args); py_s = (time.time() - t0) * 100   # extrapolate to 200k
    out = dict(numba=nb.__version__, steps=200_000, rings=NR, compile_s=round(compile_s, 2),
               numba_run_s=round(run_s, 3), pure_python_est_s=round(py_s, 1),
               speedup=round(py_s / run_s), rms_err_pm=float(rms * 1e3), median_abs_err_pm=float(med * 1e3), max_err_pm=float(mx * 1e3),
               final_heater_mW=[round(float(v), 2) for v in pfin])
    env = RingBankEnv()
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    t0 = time.time()
    for _ in range(1000):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
    out["gymnasium"] = gym.__version__
    out["gym_check_env"] = "passed"
    out["gym_steps_per_s"] = round(1000 / (time.time() - t0))
    print(json.dumps(out, indent=1))
    os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "out", "t11_closed_loop_numba_gym.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
