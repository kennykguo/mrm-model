"""Smoke test: Stable-Baselines3 PPO trains on the gymnasium RingBankEnv of t11 (a few thousand steps,
just to prove the RL stack runs; no claim about the policy). Run:
  uv run --isolated --no-project --python 3.12 --with stable-baselines3 --with numba --with gymnasium \
     python docs/research/05_tools_smoke/t14_sb3_ppo.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t11_closed_loop_numba_gym import RingBankEnv
import stable_baselines3 as sb3, torch, gymnasium as gym
from gymnasium.wrappers import TimeLimit

env = TimeLimit(RingBankEnv(), max_episode_steps=500)
t0 = time.time()
model = sb3.PPO("MlpPolicy", env, n_steps=1024, batch_size=256, verbose=0, seed=0, device="cpu")
model.learn(total_timesteps=4096)
obs, _ = env.reset(seed=1)
act, _ = model.predict(obs, deterministic=True)
print(json.dumps(dict(sb3=sb3.__version__, torch=torch.__version__, gymnasium=gym.__version__,
                      trained_steps=4096, action_shape=list(act.shape), seconds=round(time.time() - t0, 1))))
