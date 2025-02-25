# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Deep Reinforcement Learning for Power-to-Gas Dispatch Optimization
# GitHub Repository: https://github.com/SimMarkt/RL_PtG
#
# rl_tb: 
# > Starts the tensorboard server for monitoring of RL training results
# ----------------------------------------------------------------------------------------------------------------

import os

# ----------------------------------------------------------------------------------------------------------------------
print("Tensorboard URL...")
os.system('tensorboard --logdir=tensorboard/')