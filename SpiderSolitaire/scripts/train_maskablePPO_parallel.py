# this script runs thi]
import os

import json
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
    
class SpiderParallelCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        env = self.model.get_env()
        
        # Get last dones and infos from the environment
        # These are accessible via the vectorized environment's internal tracking
        try:
            infos = self.model.env.env_method("get_info") if hasattr(self.model.env, 'env_method') else []
            seqs = [info.get("n_complete_sequences", -1) for info in infos]
            facedown_cards = [info.get("n_facedown_tableau_cards", -1) for info in infos]

            if self.verbose > 0:
                for i, (s, f) in enumerate(zip(seqs, facedown_cards)):
                    print(f"Env {i}: completed sequences: {s}, facedown cards: {f}")

            self.logger.record("spider/mean_complete_sequences", np.mean(seqs))
            self.logger.record("spider/mean_facedown_tableau_cards", np.mean(facedown_cards))
            self.logger.record("spider/std_complete_sequences", np.std(seqs))
            self.logger.record("spider/std_facedown_tableau_cards", np.std(facedown_cards))
        except Exception as e:
            if self.verbose > 0:
                print(f"SpiderParallelCallback: Could not record metrics: {e}")
        
        return True
    
import sb3_contrib
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sb3_contrib.common.wrappers import ActionMasker


def train_maskable_ppo_parallel(n_envs: int=CPU_CORES_DEFAULT_LIMIT,
                                n_suits: int =1, log_dir: str = None, save_dir: str=None,

                                actions_limit: int =1000, n_steps: int =1*1000, batch_size: int =1000, total_timesteps: int =1000_000,
                                verbose: int =1,
                                lr: float =3e-4, gamma: float =0.995,
                                rewards_policy:dict[str, float]={"discover_card": 4,
                                                                    "free pile": 6,
                                                                    'extend sequence': 0.75,
                                                                    "deal cards": -0.5},
                                    _render_state_timeout=1_0000,  _diagnostics_mode=0, 
                                    tb_log_name: str =None
                                ):
    """Train MaskablePPO on Spider Solitaire environment."""
    def make_env(rank, seed=0):
        def _init():
            env = SpiderEnv(n_actions_limit=actions_limit, _dtype=np.float32, _render_state_timeout=_render_state_timeout, 
                                _diagnostics_mode=_diagnostics_mode, vectorize_obs=(n_suits>1), mask_legal_actions=True,
                            #rewards_limit=True,
                            n_suits=n_suits,
                            
                            rewards_policy =rewards_policy
                            )  
            env = ActionMasker(env, action_mask_fn=lambda e: e.unwrapped.get_action_mask())
            env.reset(seed=seed+rank)
            return env
        return _init
    
    if log_dir is None:
        log_dir = LOG_DIR+f'MaskablePPO_{n_suits}suit/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if save_dir is None:
        save_dir = SAVE_MODELS_DIR+f'MaskablePPO_{n_suits}suit/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_max_envs =  ncores if (ncores:=os.cpu_count()) is not None else 1

    if n_envs > n_max_envs:
        print(f"NOTE: Requested n_envs={n_envs} exceeds number of CPU cores={n_max_envs}. Forced to use n_envs={n_max_envs}.")
        n_envs = n_max_envs

    print(f"Using {n_envs} environments in parallel, each with action limit {batch_size}, total action limit per game: {actions_limit}")

    # Create parallel environments:
    envs = [make_env(i) for i in range(n_envs)]
    #should be wrapped by __main__ block 
    envp = SubprocVecEnv(envs)
    envp = VecMonitor(envp,filename=log_dir+"vecmonitor.csv")

    rew_code = ''.join('_'+k[0]+(k[k.rfind(' ')+1] if k.rfind(' ') else '')+str(round(v,1))  for k, v in rewards_policy.items() )
    model_alias = f"MaskablePPO_{n_suits}suits_{n_envs}_parenvs_{actions_limit}_{n_steps}_{batch_size}_{rew_code}"
    l = model.learn(total_timesteps=total_timesteps,callback=SpiderParallelCallback(), 
                    tb_log_name=model_alias if tb_log_name is None else tb_log_name)
    model.save(save_dir+model_alias)
    return model

if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_envs', type=int, default=CPU_CORES_DEFAULT_LIMIT)
    parser.add_argument('--n_suits', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--actions_limit', type=int, default=1000)
    parser.add_argument('--n_steps', type=int, default=4*1000//round(np.power(CPU_CORES_DEFAULT_LIMIT, 1/3))-1)      
    # rollout_size = n_steps * n_envs
    parser.add_argument('--batch_size', type=int, default=1000) # must divide rollout_size
    parser.add_argument('--total_timesteps', type=int, default=1_000_000//round(np.power(CPU_CORES_DEFAULT_LIMIT, 1/3)))

    parser.add_argument('--verbose', type=int, default=1)   

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.995)

    parser.add_argument('--rewards_policy', type=str, default="{'discover_card': 4,  'free pile': 8, 'extend sequece': 0.7, 'deal cards': -0.5}")

    parser.add_argument('--_render_state_timeout', type=int, default=1_0000)
    parser.add_argument('--_diagnostics_mode', type=int, default=0)
    parser.add_argument('--tb_log_name', type=str, default=None)
    args = parser.parse_args()

    # Parse rewards_policy robustly: accept JSON (double quotes) or Python dict literal (single quotes)
    import ast
    try:
        rewards_policy_parsed = json.loads(args.rewards_policy)
    except Exception:
        rewards_policy_parsed = ast.literal_eval(args.rewards_policy)

    model = train_maskable_ppo_parallel(n_envs=args.n_envs, n_suits=args.n_suits, log_dir=args.log_dir,
                                        actions_limit=args.actions_limit, n_steps=args.n_steps, batch_size=args.batch_size, total_timesteps=args.total_timesteps,
                                        verbose=args.verbose, lr=args.lr, gamma=args.gamma,
                                        rewards_policy=rewards_policy_parsed,
                                        
                                        _render_state_timeout=args._render_state_timeout,  _diagnostics_mode=args._diagnostics_mode,
                                        tb_log_name=args.tb_log_name)
        