# my personal note:
# activate venv `myenv` from conda powershell: conda activate C:\Users\alex3\.conda_envs\rl_env
# then run this srcript:                       python .\SpiderSolitaire\scripts\train_maskablePPO_multiple_configs_parallel.py
# 
 
import math
import os

import json
import time
import torch

import sys
from pathlib import Path
# Ensure repository root (RL) is on sys.path regardless of current working directory
# This makes imports stable when running the script from different locations.
# project root is two levels up from this script (RL)
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
from SpiderSolitaire.env.spider_env import SpiderEnv


LOG_DIR = "./SpiderSolitaire/logs/training/"
SAVE_MODELS_DIR = "./SpiderSolitaire/models/"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CPU_CORES_DEFAULT_LIMIT = ncores if os.cpu_count() is not None and (ncores:=int(0.9*os.cpu_count())) else 1



from stable_baselines3.common.callbacks import BaseCallback
    
from stable_baselines3.common.callbacks import BaseCallback

class SpiderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:   
        env = self.model.get_env()
        info = env.env_method("get_info")[0]

        seqs = info.get("n_complete_sequences", 0)
        facedown_cards = info.get("n_facedown_tableau_cards", 0)
        self.logger.record("spider/complete_sequences", seqs)
        self.logger.record("spider/facedown_tableau_cards", facedown_cards)

        return True
    
import sb3_contrib
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sb3_contrib.common.wrappers import ActionMasker


def worker(agent_configs_list, env_configs_list, model_aliases_list, worker_rank:int=0, total_timesteps: int=1_000_000, 
           save_dir: str=None, log_dir: str=None, env_actions_limit: int=1000, mask_legal_actions=True, 
           n_suits: int=1, default_learning_rate: float=3e-4, default_gamma: float=0.995):
    """Worker function: creates environments and models, trains them locally."""
    
    for i, (agent_config, env_config, model_alias) in enumerate(zip(agent_configs_list, env_configs_list, model_aliases_list)):
        # Create environment locally
        env = ActionMasker(
            Monitor(
                SpiderEnv(
                    n_actions_limit=env_actions_limit, 
                    mask_legal_actions=mask_legal_actions, 
                    vectorize_obs=(n_suits > 1),
                    n_suits=n_suits,
                    **env_config
                ), 
                filename=log_dir + model_alias + "_monitor.csv"
            ),
            action_mask_fn=lambda e: e.unwrapped.get_action_mask()
        )
        
        # Create model locally
        model = MaskablePPO(
            "MlpPolicy", 
            env, 
            device="auto",
            tensorboard_log=agent_config.get('tensorboard_log', log_dir),
            learning_rate=agent_config.pop('learning_rate', default_learning_rate),
            gamma=agent_config.pop('gamma', default_gamma),
            verbose=0,
            **agent_config
        )
        
        # Train
        timesteps = agent_config.get('total_timesteps', total_timesteps)
        print(f"Worker {worker_rank}, training during {timesteps} timesteps {i+1}/{len(agent_configs_list)}: {model_alias}...")
        model.learn(total_timesteps=timesteps, callback=SpiderCallback(), tb_log_name=model_alias)
        model.save(save_dir + model_alias)

def train_maskable_ppo_parallel(agent_configs: dict[str, int], env_configs: dict[str, int|float|dict[str, float]], 
                                total_timesteps: int=1_000_000, env_actions_limit: int=1000, mask_legal_actions=True,
                                n_processes: int=CPU_CORES_DEFAULT_LIMIT,
                                n_suits: int =1,
                                default_learning_rate: float =3e-4, default_gamma: float =0.995,
                                
                                log_dir:str=None, __log_dir: str =LOG_DIR+"MaskablePPO_multiple_configs_parallel/",
                                save_dir: str=None, __save_dir: str=SAVE_MODELS_DIR+"MaskablePPO_multiple_configs_parallel/", 
                                verbose: int =0
                                ):
    """Train MaskablePPO on Spider Solitaire environment with different configs running in parallel. 
    
    If number of env configs is less than number of agent configs, the env configs are broadcasted to match the number of agent configs: env configs will alternate sequentially among the agent configs. 
    If number of env configs matches number of agent configs, they are paired one-to-one.
    In number of env configs exceeds number of agent configs, 
    """
    assert len(agent_configs) >0 and len(env_configs) >0, "At least one agent config and one environment config must be provided."
    assert all('n_steps' in cfg for cfg in agent_configs), "Each agent config must specify 'n_steps'."
    assert all('batch_size' in cfg for cfg in agent_configs), "Each agent config must specify 'batch_size'."


    if len(env_configs)<len(agent_configs):
        # Broadcast single env config to all agent configs
        n_repeats=math.ceil(len(agent_configs)/len(env_configs))
        print(f"Broadcasting {len(env_configs)} environment configs to match {len(agent_configs)} agent configs. At least one env config will be used {n_repeats} times.\n")
        env_configs = env_configs * n_repeats
        env_configs = env_configs[:len(agent_configs)] 

    elif len(env_configs)>len(agent_configs):
        # Broadcast agent configs to match env configs
        n_repeats=math.floor(len(env_configs)/len(agent_configs))
        print(f"Broadcasting {len(agent_configs)} environment configs to match {len(env_configs)} agent configs. At least one env config will be used {n_repeats} times.\n")
        agent_configs =  agent_configs * n_repeats
        agent_configs = agent_configs[:len(env_configs)]

    if log_dir is None:
        print(f"No log_dir provided, using default {__log_dir}")
        log_dir = __log_dir
    if save_dir is None:    
        print(f"No save_dir provided, using default {__save_dir}")
        save_dir = __save_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    from multiprocessing import Process, Queue

    processes = []
    #queue = Queue()

    n_processes = min(n_processes, len(agent_configs), len(env_configs), os.cpu_count() if os.cpu_count() is not None else 1)
    time0 = time.time()
    print(f"Starting training with {n_processes} parallel processes...")
    for i_process in range(n_processes):
        process_agent_configs = [cfg for i, cfg in enumerate(agent_configs) if i % n_processes == i_process]
        process_env_configs   = [cfg for i, cfg in enumerate(env_configs)   if i % n_processes == i_process]

        special_codes = {'discover card': 'dd'}
        rew_codes = [''.join('_'+ ((k[0]+(k[k.rfind(' ')+1] if k.rfind(' ') else '')) if k not in special_codes else special_codes[k]) + str(round(v,5))  for k, v in config['rewards_policy'].items() ) 
                        for config in process_env_configs]
        agent_cfg_codes = [f"{config['n_steps']}_{config['batch_size']}_{config.get('gamma', default_gamma)}_{config.get('lr', default_learning_rate)}" for config in process_agent_configs]   

        model_aliases =  [f"MaskablePPO_{n_suits}suits_{env_actions_limit}_{agent_cfg_code}_{rew_code}" for agent_cfg_code, rew_code in zip(agent_cfg_codes, rew_codes)]
        
        # Pass only config dicts, not environment/model instances (for Windows pickle compatibility)
        p = Process(
            target=worker, 
            args=(process_agent_configs, process_env_configs, model_aliases), 
            kwargs={
                'worker_rank': i_process, 
                'total_timesteps': total_timesteps, 
                'save_dir': save_dir, 
                'log_dir': log_dir,
                'env_actions_limit': env_actions_limit,
                'mask_legal_actions': mask_legal_actions,
                'n_suits': n_suits,
                'default_learning_rate': default_learning_rate,
                'default_gamma': default_gamma
            }
        )
        processes.append(p)
        p.start()

    #rewards, seqlens, facedown_cards = [], [], []

    for p in processes:
        p.join()
    
    # while not queue.empty():
    #     res = queue.get()
    time1 = time.time()
    print(f"All training processes completed in {time1 - time0:.2f} seconds.")

if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--configs_file_path', type=str, required=False,
                             help="Path to JSON file containing list of agent and environment configs.", 
                             default=None)
    parser.add_argument('--n_processes', type=int, default=os.cpu_count() if os.cpu_count() is not None else 1)
    parser.add_argument('--n_suits', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--actions_limit', type=int, default=1024)
    parser.add_argument('--total_timesteps', type=int, default=1024*1024,
                        help="Total timesteps per agent config if not specified in the config.")

    parser.add_argument('--verbose', type=int, default=0)   

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.995)

    args = parser.parse_args()

    if args.configs_file_path is None:
        config_filepath = './SpiderSolitaire/scripts/ppo_multiple_configs.json'
        print(f"No configs file path provided, using default {config_filepath}")
    else:
        config_filepath = args.configs_file_path

    # # Parse rewards_policy robustly: accept JSON (double quotes) or Python dict literal (single quotes)
    # import ast
    # try:
    #     rewards_policy_parsed = json.loads(args.rewards_policy)
    # except Exception:
    #     rewards_policy_parsed = ast.literal_eval(args.rewards_policy)

    with open(config_filepath, 'r', encoding='utf-8') as f:
        configs = json.load(f)
    model = train_maskable_ppo_parallel(configs['agent_configs'], configs['env_configs'],
                                        total_timesteps=args.total_timesteps, env_actions_limit=args.actions_limit,
                                        n_processes=args.n_processes, n_suits=args.n_suits,
                                        default_learning_rate=args.lr, default_gamma=args.gamma,
                                        log_dir=args.log_dir, save_dir=args.save_dir,
                                        verbose=args.verbose)
        