# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Deep Reinforcement Learning for Power-to-Gas dispatch optimization
# https://github.com/SimMarkt/RL_PtG

# rl_config_train: 
# > Contains the configuration and settings for RL training 
# > Converts the config_train.yaml data into a class object for further processing
# ----------------------------------------------------------------------------------------------------------------

import yaml

class TrainConfiguration:
    def __init__(self):
        # Load the environment configuration
        with open("config/config_train.yaml", "r") as env_file:
            train_config = yaml.safe_load(env_file)

        # Unpack data
        self.__dict__.update(train_config)
        
        # Set configuration
        self.path = None                        # RL_PtG folder path
        self.slurm_id = None                    # SLURM ID of a specific thread
        com_set = ['pc', 'slurm']
        
        assert self.com_conf in com_set, f"Wrong computation setup specified - data/config_train.yaml -> com_conf : {self.com_conf} must match {com_set}"
        train_set = ['simple_train', 'save_model', 'load_model', 'save_load_model']
        assert self.model_conf in train_set, f"Wrong training setup specified - data/config_agent.yaml -> model_conf : {train_config['model_conf']} must match {train_set}"

        assert len(self.r_seed_train) == len(self.r_seed_test), 'Number of random seeds must be equal for the training and test set!'
        self.seed_train = None              # Random training seed of the present thread
        self.seed_test = None               # Random validation/test seed of the present thread







       



