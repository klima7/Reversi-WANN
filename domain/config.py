from collections import namedtuple
import numpy as np

Game = namedtuple('Game', ['env_name', 'time_factor', 'actionSelect',
  'input_size', 'output_size', 'layers', 'i_act', 'h_act',
  'o_act', 'weightCap','noise_bias','output_noise','max_episode_length','in_out_labels'])

games = {}

cartpole_swingup = Game(env_name='CartPoleSwingUp',
  input_size=5,
  output_size=1,
  in_out_labels=['x','x_dot','cos(theta)','sin(theta)','theta_dot', 'force'],
  actionSelect='all',
  layers=[5, 5],
  i_act=np.full(5,1),
  o_act=np.full(1,5),
  h_act=[1,2,3,4,5,6,7,8,9,10],
  time_factor=0,
  weightCap = 2.0,
  noise_bias=0.0,
  output_noise=[False, False, False],
  max_episode_length=1000,
)
games['swingup'] = cartpole_swingup


small_reversi = Game(env_name='small_reversi',
  input_size=36,
  output_size=36,
  in_out_labels=[f'{i}' for i in range(36)] * 2,
  actionSelect='softmax',
  layers=[5, 5],  # anything

  i_act=np.full(5,1), # (input_size, 1)
  o_act=np.full(1,5), # (output_size, 1)
  h_act=[1,2,3,4,5,6,7,8,9,10], # (step functions)


  time_factor=0,
  weightCap = 2.0,
  noise_bias=0.0,
  output_noise=[False, False, False],
  max_episode_length = 1000,
)
games['small_reversi'] = small_reversi
