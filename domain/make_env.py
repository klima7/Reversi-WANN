import numpy as np
import gym


def make_env(env_name, seed=-1, render_mode=False):
  if "Bullet" in env_name:
    import pybullet as p
    import pybullet_envs
    import pybullet_envs.bullet.kukaGymEnv as kukaGymEnv

  # -- Cart Pole Swing up -------------------------------------------- -- #
  if (env_name.startswith("CartPoleSwingUp")):
    from domain.cartpole_swingup import CartPoleSwingUpEnv
    env = CartPoleSwingUpEnv()


  # -- Other  -------------------------------------------------------- -- #
  else:
    env = gym.make(env_name)

  if (seed >= 0):
    domain.seed(seed)

  return env