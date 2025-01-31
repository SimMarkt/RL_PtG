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
        
        # Set configuration
        self.path = None                        # RL_PtG folder path
        com_set = ['pc', 'slurm']
        self.slurm_id = None                                # SLURM ID of a specific thread
        assert train_config['com_conf'] in com_set, f"Wrong computation setup specified - data/config_train.yaml -> com_conf : {train_config['com_conf']} must match {com_set}"
        self.com_conf = train_config['com_conf']               # selected computational resources either 'pc' or 'slurm'
        self.device = train_config['device']               # computational device ['cpu', 'gpu', 'auto']
        self.str_inv = train_config['str_inv']        # specifies the training results and models to a specific investigation ##################---------------------------------
        self.str_inv_load = train_config['str_inv_load']  # specifies the name of the pretrained model  
        train_set = ['simple_train', 'save_model', 'load_model', 'save_load_model']
        assert train_config['model_conf'] in train_set, f"Wrong training setup specified - data/config_agent.yaml -> model_conf : {train_config['model_conf']} must match {train_set}"
        self.model_conf = train_config['model_conf']   # simple_train: Train RL from scratch without saving the model afterwards
                                    # save_model: Train RL from scratch and save the model afterwards
                                    # load_model: Load a pretrained RL model and continue with training without saving the model afterwards
                                    # save_load_model: Load a pretrained RL model and continue with training and save the model afterwards
        self.path_files = train_config['path_files']   # data path with pretrained RL models

        self.parallel = train_config['parallel']   # specifies the computation setup: "Singleprocessing" (DummyVecEnv) or "Multiprocessing" (SubprocVecEnv)
        self.train_or_eval = train_config['train_or_eval']         # specifies whether the environment provides detailed descriptions of the state for evaluation ("eval") or not ("train" - recommended for training)
        self.eval_trials = train_config['eval_trials']              # No. of evaluation trials in validation and testing (to appropriately assess stochastic policies)

        self.train_steps = train_config['train_steps']          # No. of training steps
        self.test_steps = train_config['test_steps']            # Validation interval (Number of steps after which the RL agent is evaluated)
        self.r_seed_train = train_config['r_seed_train']         # random seeds for neural network initialization (and environment randomness) of the training set
        self.r_seed_test = train_config['r_seed_test']         # random seeds for neural network initialization (and environment randomness) of the validation and test sets
        assert len(self.r_seed_train) == len(self.r_seed_test), 'Number of random seeds must be equal for the training and test set!'
        self.seed_train = None              # Random training seed of the present thread
        self.seed_test = None               # Random validation/test seed of the present thread







       



