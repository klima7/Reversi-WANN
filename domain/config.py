from collections import namedtuple
import numpy as np

Game = namedtuple('Game', ['env_name', 'time_factor', 'actionSelect',
  'input_size', 'output_size', 'layers', 'i_act', 'h_act',
  'o_act', 'weightCap','noise_bias','output_noise','max_episode_length','in_out_labels'])

games = {}

cartpole_swingup = Game(env_name='CartPoleSwingUp',
  actionSelect='all', # all, soft, hard
  input_size=5,
  output_size=1,
  time_factor=0,
  layers=[5, 5],
  i_act=np.full(5,1),
  h_act=[1,2,3,4,5,6,7,8,9,10],
  o_act=np.full(1,5),
  weightCap = 2.0,
  noise_bias=0.0,
  output_noise=[False, False, False],
  max_episode_length = 1000,
  in_out_labels = ['x','x_dot','cos(theta)','sin(theta)','theta_dot', 'force']
)
games['swingup'] = cartpole_swingup
