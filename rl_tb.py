# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Deep Reinforcement Learning for Power-to-Gas dispatch optimization
# https://github.com/SimMarkt/RL_PtG

# rl_tb: 
# > Returns the URL of the tensorboard web server for monitoring of RL training
# ----------------------------------------------------------------------------------------------------------------

import os

# # ----------------------------------------------------------------------------------------------------------------------
print("Tensorboard URL...")
os.system('tensorboard --logdir=' + 'tensorboard/')