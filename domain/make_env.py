def make_env(env_name, seed=-1, render_mode=False):
  if (env_name.startswith("CartPoleSwingUp")):
    from domain.cartpole_swingup import CartPoleSwingUpEnv
    env = CartPoleSwingUpEnv()

  elif (env_name.startswith("small_reversi")):
    from domain.reversi import SmallReversiEnv
    env = SmallReversiEnv()

  if (seed >= 0):
    domain.seed(seed)

  return env