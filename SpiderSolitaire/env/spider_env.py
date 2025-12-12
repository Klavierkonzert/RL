import typing
import numpy as np
import numpy.typing as npt
import gymnasium as gym 
from numba import njit

from IPython.display import clear_output

import sys
from pathlib import Path
# adjust '..' to point to your repository root that contains the SpiderSolitaire package
sys.path.insert(0, str(Path('..').resolve()))

from utils.io import timed_input
from env.spider_space import SpiderSpace

class SpiderEnv(gym.Env):
    """Introduction (Wikipedia):
    
        The main purpose of the game is to remove all cards from the table, assembling them in the tableau before removing them.
        Initially, 54 cards are dealt to the tableau in ten piles, face down except for the top cards. 
        The tableau piles build down by rank, and in-suit sequences can be moved together. 
        The 50 remaining cards can be dealt to the tableau ten at a time when none of the piles are empty.

        A typical Spider layout requires the use of two decks. 
        The tableau consists of 10 stacks, with 6 cards in the first 4 stacks, with the 6th card face up, and 5 cards in the remaining 6 stacks, with the 5th card face up. 
        Each time the stock is used, it deals out one card to each stack.
    """
    
    N_PILES = SpiderSpace.N_PILES # +1 for undealt cards
    N_TARGETS = SpiderSpace.N_TARGETS
    NONCARD_VALUE = SpiderSpace.NONCARD_VALUE
    #metadata = {"render.modes": ["human"]}

    DEFAULT_REWARDS = {"complete sequence": 100.0,
                       "victory": N_PILES*100.0,
                       'step':-0.2,
                       "NOOP": -0.4,
                      "discover card": 8.0,
                      "extend sequence": 2.0,
                      "deal cards": 0.,
                      "free pile": 32,
                      'no available actions': -100,
                       "reward limit": 200
                      }
    # no needto keep this dict, as only one limit matters (for extending sequences)
    # DEFAULT_REWARDS_LIMITS = {"complete sequence": 100.0,
    #                   "discover card": 8.0 *SpiderSpace.N_FACEDOWN_CARDS,
    #                   "extend sequence": N_TARGETS*(SpiderSpace.HIGHEST_RANK - SpiderSpace.LOWEST_CARD)*DEFAULT_REWARDS["extend sequence"],
    #                   "deal cards": 0.
    #                          }

    MOVE_ACTIONS_RANGE = N_PILES*(N_PILES)*(SpiderSpace.HIGHEST_RANK -1)
    DEAL_CARDS_ACTION = MOVE_ACTIONS_RANGE
    NOOP_ACTION = DEAL_CARDS_ACTION +1 # It's possible in Spider Solitaire game that there are no available actions (game over)
    ACTIONS_RANGE = NOOP_ACTION+1
    
    def __init__(self, n_suits: int=4, n_actions_limit:int=None, 
                 vectorize_obs: bool=False,
                 mask_legal_actions:bool=False,
                 rewards_policy: dict[str, float]=None,
                 rewards_limits:dict[str, float]=None,
                 _render_state_timeout:int=None, _diagnostics_mode:bool=False,
                 _maxsize: int = 64, _dtype=np.int8):
        """"...
        :params:
            n_suits: int=4 - number of suits in the game. Popular options are 1, 2 and 4 (can be thought as difficulty levels)
            n_actions_limit:int=None - maximum number of actions an agent can perform in single game before it's restarted and a new game is generated
            vectorize_obs: bool=False - if True, `get_obs` method yields observations in 3D-form array (thus each card is represented by a [0,...,rank,..,0] 4-vector), this does not affect interna enviroment calculations which are done in 2D (i.e. each card is represented by an integer)
            mask_legal_actions: bool=False - if True, each `step` a mask for legal moves is computed within `get_info` method
                 
            _render_state_timeout:int=None 
            _maxsize: int = 64, _dtype=np.int8
        """
        self.observation_space = SpiderSpace(n_suits=n_suits, maxsize=_maxsize, 
                                             vectorize=vectorize_obs, dtype=_dtype)
        # possible actions are: 1) move a sequence of cards from pile A to pile B on the table (the length can be 1,...,12), so 10*9*12 options
        # 2) deal 10 cards from stock pile, this is the last action
        self.action_space = gym.spaces.discrete.Discrete(SpiderEnv.ACTIONS_RANGE)

        self._internal_state, self._tableau_piles, self._stock_pile = None,None,None
        # self.facedown_cards, self.stock_cards = [], -1
        
        self._complete_sequences = 0#{i:0 for i in range(n_suits)} # Agent's goal is to complete all suits
        
        # n_actions_limit could be necessary to limit number of episodes in a game
        self._actions_limit = n_actions_limit
        self._actions_counter=0
            
        self._stock_len = SpiderSpace.N_STOCK_CARDS
        self._facedown_tableu_cards=SpiderSpace.N_FACEDOWN_CARDS

        self._render_state_timeout, self._diagnostics_mode =_render_state_timeout, _diagnostics_mode

        self.episode_over = False

        self.mask_legal_actions = mask_legal_actions

        if rewards_policy is not None:
            self.rewards_policy = {**SpiderEnv.DEFAULT_REWARDS ,**rewards_policy}
        else:
            self.rewards_policy = SpiderEnv.DEFAULT_REWARDS 

        if rewards_limits is not None:
            # if set, rewards are extracted every time from these limits
            self.rewards_limits=rewards_limits

    def reset(self,seed: int|None=None, options=None,
              __N_STOCK_CARDS = SpiderSpace.N_STOCK_CARDS, __N_FACEDOWN_CARDS=SpiderSpace.N_FACEDOWN_CARDS
              ) -> tuple[npt.NDArray[np.integer], dict[str,typing.Any]]:
        """resets the environment to initial state and randomly generates a new game"""
        super().reset(seed=seed) # np_seed

        self.episode_over = False
        
        self._actions_counter = 0
        self._complete_sequences = 0
        
        self.observation_space.seed(seed)
        self._internal_state = self.observation_space.sample(hide_cards = True)
        
        self._tableau_piles, self._stock_pile, self.counts, self.seq_depths = self._unpack_state(self._internal_state)
        
        self._stock_len = __N_STOCK_CARDS
        self._facedown_tableu_cards=__N_FACEDOWN_CARDS
        
        #initial visible observation
        obs = self.get_obs()
        info = self.get_info()
        return  obs, info #(obs, info) if return_info else obs
        
    #@staticmethod
    #@njit(inline='always')
    def get_obs(self,# _vectorize:bool=True,
                __N_PILES=N_PILES, __I_FACEDOWN_CNTS=SpiderSpace.I_FACEDOWN_CNTS, __DTYPE=SpiderSpace.DTYPE) -> npt.NDArray[np.integer]:
        """Returns the observation which is available to an agent (i.e. facedown cards are masked and piles are left-aligned)"""
        #last feature (row, pile) -> num of facedown cards (6,6,6,6,5,5,..., 50) in the beggining of a game 
        #piles 0,...,N_PILES-1 -> tableau piles
        #pile N_PILES -> stock pile
        obs_state = self._internal_state
        if len(self.observation_space.shape)==3:
            #recalcs FACEDOWN_COUNTS and DEPTH_COUNTS
            obs_state = SpiderSpace._vectorize(self._internal_state, _counts_features=True)
            # after vectorization agent can see how many facedown cards belong to each suit, 
            # this info should not be available:
            obs_state[__I_FACEDOWN_CNTS] =  obs_state[__I_FACEDOWN_CNTS].sum(axis=1)[:,None]
            
        obs_state = obs_state*(obs_state>0) #masking facedown cards
        obs_state[:__N_PILES] = SpiderSpace._left_align_rows(obs_state[:__N_PILES])
        
        if self.observation_space.dtype!=__DTYPE:
            obs_state = obs_state.astype(self.observation_space.dtype)
        return obs_state
        
    def get_info(self) -> dict[str,typing.Any]:
        d = {'n_complete_sequences' : self._complete_sequences,
            'n_facedown_tableu_cards': self._facedown_tableu_cards,
            'n_remaining_stock_cards': self._stock_len
        }
        if self.mask_legal_actions:
             d['action_mask'] = self.get_action_mask()
        return d


    def _get_reward(self, action:int, 
                    terminated, truncated,
                    tableau_piles,
                    n_completed_seqs, 
                    init_stock_cards,
                    init_facedown_cnts, facedown_counts,
                    init_depths,        seq_depths,
                    init_tableau_cards,
                    __N_TARGETS=N_TARGETS, __NONCARD_VALUE=NONCARD_VALUE, 
                    __MOVE_ACTIONS_RANGE=MOVE_ACTIONS_RANGE, __DEAL_CARDS_ACTION = DEAL_CARDS_ACTION, __NOOP_ACTION=NOOP_ACTION
                   ) -> float:

        reward= 0
        if action is not None:
            if n_completed_seqs:
                reward += self.rewards_policy['complete sequence']*(n_completed_seqs)
                    
            reward += self.rewards_policy['discover card']*(init_facedown_cnts - facedown_counts[:__N_TARGETS]).sum()
            if action<=__MOVE_ACTIONS_RANGE: #if not deal cards action
                # seq depth is increased
                reward += self.rewards_policy['extend sequence']*(init_depths - seq_depths).sum()
                
                # reward for freeing a pile
                tab_cards = np.count_nonzero(tableau_piles-__NONCARD_VALUE, axis=1)
                n_freed_piles = ((tab_cards==0) & (tab_cards-init_tableau_cards<1)).sum()
                
                ##_extend_sequence_reward coulbe used here
                #reward+= self.rewards_policy["free pile"]*n_freed_piles
                ##to inttroduce this reward, prevent an agent from moving sequences back and forth

            elif action==__DEAL_CARDS_ACTION and "deal cards" in self.rewards_policy and self.rewards_policy["deal cards"]!=0:
                # punish for dealing cards:
                coefff =  self.rewards_policy["deal cards"]
                if type(coefff)==tuple:
                    deal_cards_reward = SpiderEnv._deal_cards_reward(init_stock_cards, init_facedown_cnts, init_depths,
                                                                       *coefff )
                else:
                    deal_cards_reward = SpiderEnv._deal_cards_reward(init_stock_cards, init_facedown_cnts, init_depths,
                                                                       coeff=coefff )
                reward += deal_cards_reward
                #print debug info:
                if self._diagnostics_mode>=2 or self._diagnostics_mode>=1.5 and self._render_state_timeout is not None and self._render_state_timeout>0 :
                    print(f"Punishment/reward for dealing cards: {deal_cards_reward}")
                    self.render(print_auxillary_info=True)
                    
            ##punishment if no available actions left
            #elif  action==__NOOP_ACTION and not terminated and "no available actions" in self.rewards_policy : 
            #    reward += self.rewards_policy['no available actions'] *(__N_TARGETS-self._complete_sequences )* ((self._actions_limit-self._actions_counter)/self._actions_limit if self._actions_limit is not None else 1)
            elif  action==__NOOP_ACTION and not terminated and "no available actions" in self.rewards_policy:
                reward +=self.rewards_policy['NOOP']
            
            if 'step' in self.rewards_policy:
                reward += self.rewards_policy['step'] # punisment for a step
                
        # else:
        #     reward -= 010 #punishment for invalid action
    

            
        # #punishment for losing game
            # if truncated:
            #     reward -= (__N_TARGETS-self._complete_sequences)**2 
        # reward for victory:
        if self._complete_sequences == __N_TARGETS and 'victory' in self.rewards_policy:
            reward += self.rewards_policy['victory']

        return reward


            
    def step(self, action:int,
            __N_TARGETS=N_TARGETS, __NONCARD_VALUE=NONCARD_VALUE, 
            __NOOP_ACTION=NOOP_ACTION)-> tuple:
        """
        Executes action in the environment.
        """
        assert self._complete_sequences < __N_TARGETS #and (self._actions_limit is None or self._actions_limit is not None and self._actions_limit-self._actions_counter>=-1)

        if self.episode_over:
            obs, info = self.reset()
            if self._actions_limit is not None and 0<self._actions_limit==self._actions_counter:
                return obs, 0.0, False, True, info
            else:
                return obs, 0.0, True, False, info
        else:
            self._actions_counter+=1
    
            tableau_piles, stock_pile, facedown_counts, seq_depths = self._unpack_state(self._internal_state)

            init_stock_cards = np.count_nonzero(stock_pile-__NONCARD_VALUE)
            init_tableau_cards = np.count_nonzero(tableau_piles-__NONCARD_VALUE, axis=1)
            
            init_complete_seqs = self._complete_sequences
            init_facedown_cnts = facedown_counts[:__N_TARGETS].copy()
            init_depths = seq_depths.copy()

            #perform chosen action:
            action_res= self._num2act(action)
            # number of completed sequences. Usually 0, sometimes 1, more on rare occasions
            n_completed_seqs= self._discard_complete_sequences()
            
            truncated = self._actions_limit is not None and self._actions_limit==self._actions_counter
    
            #the game is finished once all the sequences are collected (8) or max number of actions is reached
            #done = sum(self._complete_sequences.values()) == __N_TARGETS or self.n_actions_limit is not None and self._actions_counter==self.n_actions_limit
            terminated = truncated or (self._complete_sequences == __N_TARGETS) & action==__NOOP_ACTION 

            
            # reward will be based on number of completed sequences            
            reward = self._get_reward(action if action_res else None,
                                      terminated, truncated,
                                      tableau_piles,
                                      n_completed_seqs,
                                      init_stock_cards,
                                      init_facedown_cnts, facedown_counts,
                                      init_depths,        seq_depths,
                                      init_tableau_cards)
            if reward>(self.rewards_policy["complete sequence"]//4) and self._diagnostics_mode>0.75 or reward>self.rewards_policy["complete sequence"]//2 and self._diagnostics_mode>2:
                print('action ', self._unflatten_move_action(action)if  0<=action<self.action_space.n else action)
                print('reward',  reward)
                
            if "reward limit" in self.rewards_policy and reward>self.rewards_policy["reward limit"]:
                raise Exception(f"Too large reward detected: {reward}, diagnostic data: ", terminated, truncated,tableau_piles,
                                      n_completed_seqs,
                                      init_stock_cards,
                                      init_facedown_cnts, facedown_counts,
                                      init_depths,        seq_depths,
                                      init_tableau_cards)
            #if self._diagnostics_mode>=1 and action==__NOOP_ACTION:
                
                
                
            obs = self.get_obs()
            
            info = self.get_info()
    
            if self._render_state_timeout is not None and self._render_state_timeout>0 and self._actions_counter % self._render_state_timeout==0:
                self.render(print_auxillary_info=self._diagnostics_mode>=1, show_agents_obs=self._diagnostics_mode>=2)
                print(f"terminated: {terminated}, truncated: {truncated}")
                #print(f"Vectorized state: {SpiderSpace._vectorize(self._internal_state)}")

            self.episode_over = terminated or truncated
            return obs, reward, terminated, truncated, info

    def get_game_status(self,__N_TARGETS= N_TARGETS)-> bool:
        """Returns True if game is won, False otherwise"""
        return self._complete_sequences==__N_TARGETS
        
    ##############################################
    # possible actions function 
    # each returns True if action is valie, else False

    def get_action_mask(self,
                        __RANKS_RANGE= SpiderSpace.HIGHEST_RANK - SpiderSpace.LOWEST_CARD+1,
                        __DEAL_CARDS_ACTION = DEAL_CARDS_ACTION, __NOOP_ACTION=NOOP_ACTION
                       ) -> npt.NDArray[np.bool]:
        actions_mask = np.zeros(self.action_space.n, dtype=bool)#.reshape((5,5)) #arr = np.zeros(shape=shape, dtype=bool)
        tableau_piles, stock_pile, counts, depths = self._unpack_state(self._internal_state)
        
        if len(self.action_space.shape)<=1:
            ## STEP 0: calculate which cards can be moved from nonemtpy piles to nonempty piles

            #if depths is None:
            #    depths = SpiderSpace._get_max_sequences_depths(piles)
            #print(f"Calc depths: {depths}")

            #top cards indices per eaxh pile [empty piles are omitted]
            top_idc = SpiderSpace._get_top_cards_indices(tableau_piles)
            #print(f"Top cards: {piles[top_idc]}")

            ###if taking max depths:
            cols= np.arange(tableau_piles.shape[0])
            ## masks which cards can be moved:
            mask = (cols[depths!=0] - depths[depths!=0][:,None])<0
            #print("mask: ",mask)

            #print(f"{(depths[depths!=0][None,:]*mask)}")
            acc_depths = mask.cumsum(axis=1)#mask[:, ::-1].cumsum(axis=1)[:,::-1]
            max_depth = acc_depths.max()
            acc_depths = acc_depths[:,:max_depth]
            ## (N nonepty piles, max seq len)
            #print(f"acc depths: {acc_depths}")
            #print(f"piles as is: {tableau_piles}" )

            depth_piles = tableau_piles[top_idc[0]]* mask[None,:].T[:max_depth] # each pile per depth
            ## e.g. all the piles are present at the first coordinate
            ## thepiles with a top sequencs of depth>=2 are present at the second coordinate,
            ## etc
            #print("depth piles",depth_piles)
            #print("top idc", top_idc[1]) 
            idc_base_seq_cards = top_idc[1][:,None]-acc_depths+1

            #print(idc_base_seq_cards)

            depth_ax, pile_ax = np.arange(max_depth)[None,:], np.arange(depth_piles.shape[1])[:,None]
            base_seq_cards = depth_piles[depth_ax, pile_ax, idc_base_seq_cards].T #- piles[top_idc]
            #print(f"base seq cards:\n {base_seq_cards}")
        
            ##src_trgt_diff= base_seq_cards[:,:,None] - base_seq_cards[:,None]
            #src_trgt_diff =tableau_piles[top_idc][:,None] - base_seq_cards[:,None]
            ##depth_piles[depth_ax, pile_ax] - 
        
            #print(src_trgt_diff)
            #idepths, trgt, src = np.nonzero(src_trgt_diff ==1)
            idepths, trgt, src = np.nonzero(SpiderSpace.is_sequential(moving_card     =base_seq_cards[:,None], 
                                                                           destination_card=tableau_piles[top_idc][:,None]))
            ## src -> trgt
            #print("depths-1, targets, src piles: ", idepths, trgt, src)#top_idc[0][src], top_idc[0][trgt], idepth )
        
            ## calcularing actions indices
            ## converting to original cooords:

            #check whether not same src and trgt &
            #check whether max pile height will not be exceeded
            msk = ((src!= trgt )&  (top_idc[1][trgt]+idepths+1<tableau_piles.shape[1]) )
            
            valid_moves_idc = SpiderEnv._flatten_move_action(top_idc[0][src[msk]], 
                                                             top_idc[0][trgt[msk]], 
                                                             (idepths+1)[msk])
            #source*__N_PILES * __MAX_SEQ_LEN + target * __MAX_SEQ_LEN + depth-1
        
            actions_mask[valid_moves_idc] = True
        
            
            # STEP 2: True if possible to move cards from nonemtpy piles to empty piles:
            if top_idc[0].shape[0]< tableau_piles.shape[0]:
                empty_piles_mask = np.ones(tableau_piles.shape[0],  dtype=bool)
                empty_piles_mask[top_idc[0]] = False
                empty_piles_idc = np.nonzero(empty_piles_mask)[0]
                ##print("empty piles idc", empty_piles_idc)
                ##src_trgt_diff0 = tableau_piles[empty_piles_idc][:, None] - base_seq_cards[:, None]
                ##print("empty piles <- seqs diffs", src_trgt_diff0)
            
                ##idepths0, trgt0, src0 = np.nonzero(src_trgt_diff0<0)
                idepths0, src0 = np.nonzero(base_seq_cards)
                n_src0, n_trgt0 = src0.shape[0], empty_piles_idc.shape[0]
                idepths0, orig_src0 = np.tile(idepths0, n_trgt0), np.tile(top_idc[0][src0], n_trgt0)
                trgt0 = np.repeat(empty_piles_idc, n_src0)
                #print(trgt0, idepths0, orig_src0)
                ##print(idepths0, empty_piles_idc[trgt0], top_idc[0][src0])
                valid_moves0_idc = SpiderEnv._flatten_move_action(orig_src0,
                                                             trgt0,
                                                             idepths0+1)
        
                actions_mask[valid_moves0_idc] = True
        
            else:#STEP 3: calculate whether it is possible to deal cards [it's the last move] 
                actions_mask[__DEAL_CARDS_ACTION] = counts[-1]>0 and np.all(top_idc[0]<tableau_piles.shape[1]-1)
            #always available:
            actions_mask[__NOOP_ACTION] = True
            #print("idc of valid moves: ", np.nonzero(actions_mask))
                
            return actions_mask
        else:
            raise NotImplementedError(f'Action space shape is multidimensional')
            
    def _sample_valid_action(self):
        """Samples a valid random action"""
        return np.random.choice(np.nonzero(self.get_action_mask())[0])
        
    # @njit(inline='always')
    def _num2act(self, action: int, 
        __MOVE_ACTIONS_RANGE=MOVE_ACTIONS_RANGE, __DEAL_CARDS_ACTION = DEAL_CARDS_ACTION,
        __N_PILES=N_PILES) -> bool: # -> Callable[[SpiderEnv, int, int], int]:
        """Converts an action number into action code, performs it and returns True if action vas valid, False otherwise"""
        assert 0<=action<=self.action_space.n, "Check action range"
        if action // __MOVE_ACTIONS_RANGE==0:
            return self._move_cards_sequence(*SpiderEnv._unflatten_move_action(action))
        elif action==__DEAL_CARDS_ACTION:
            return self._deal_cards()
        else:#__NOOP_ACTION
            return False
        
    @staticmethod
    @njit(inline='always')
    def _unflatten_move_action(action: int, 
                              __MAX_SEQ_LEN=SpiderSpace.HIGHEST_RANK-1,
                              __N_PILES=N_PILES) -> tuple[int]:
        """returns (source, target, depth) value when moving cards. 
        In case source==target, the move is illegal. 
        Only `top depth`<=12 cards of the sequence are movable, these `depth` cards must be ordered and have the same suit"""
        return action // (__N_PILES * __MAX_SEQ_LEN),  (action % (__N_PILES * __MAX_SEQ_LEN))// __MAX_SEQ_LEN,  action % __MAX_SEQ_LEN +1
    @staticmethod
    @njit(inline='always')
    def _flatten_move_action(source:int, target:int, depth:int,
                              __MAX_SEQ_LEN=SpiderSpace.HIGHEST_RANK-1,
                              __N_PILES=N_PILES) -> tuple[int]:
        """returns `source*N_PILES*(HIGHEST_RANK-1) + target*(HIGHEST_RANK-1) + depth-1` value when moving cards. 
            The result of this function is inverse of `_unflatten_move_action`"""
        return source*__N_PILES * __MAX_SEQ_LEN + target * __MAX_SEQ_LEN + depth-1
   

    @staticmethod 
    #@njit(inline='always', nopython=False) # cannot determine SpiderSpace name
    def _update_depths(tableau_piles: npt.NDArray[np.integer]| list[list[int]], seq_depths: npt.NDArray[np.integer]| list[int],
        target_pile:int, trgt_height:int, source_pile:int, depth: int
        ):
        """Updates depths of two piles when moving cards sequence from one pile to another"""
        if trgt_height>0 and SpiderSpace.get_suit(tableau_piles[target_pile, trgt_height-1])==SpiderSpace.get_suit(tableau_piles[target_pile, trgt_height]):
            seq_depths[target_pile] += depth
        else:
            seq_depths[target_pile] = depth
            
        #requires updated game state (all the piles have a revealed top card)
        seq_depths[source_pile] = SpiderSpace._get_max_sequences_depths(tableau_piles[source_pile])

    @staticmethod
    @njit(inline='always')
    def _reveal_topcard(tableau_piles: npt.NDArray[np.integer]| list[list[int]], facedown_counts: npt.NDArray[np.integer]| list[int], 
                        idx_pile: int, pile_computed_topcard_idx: int,
                        __NONCARD_VALUE=NONCARD_VALUE, __HIDE_VALUE=SpiderSpace.HIDE_VALUE) -> bool:
        """Reveals a top card when previous topcard was moved. 
        Returns True (a new card was revealed) or False /num of revealed cards (1 or 0 is expected)/"""
        if tableau_piles[idx_pile, pile_computed_topcard_idx] <__NONCARD_VALUE:
            tableau_piles[idx_pile, pile_computed_topcard_idx] -= __HIDE_VALUE # revealing the next top card
            facedown_counts[idx_pile] -= 1
            return True#type(idx_pile)==int or len(idx_pile)
        else: # either empty or top card is face up:
            #do nothing
            return False
        
    @staticmethod
    # @njit(inline='always', nopython=False)
    def _swap_cards(tableau_piles: npt.NDArray[np.integer]| list[list[int]], 
        source: int, target: int, depth: int, src_height:int , trgt_height: int, 
        facedown_counts: npt.NDArray[np.integer]| list[int]=None, seq_depths: npt.NDArray[np.integer]| list[int]=None, 
        __NONCARD_VALUE=NONCARD_VALUE) -> bool:
        """Swaps `depth` cards from `source` pile to `target` pile at specified heights.

            :params:
                - tableau_piles: game state tableau piles\n
                - source: idx of a pile from which cards are taken\n
                - target: idx of a pile - destination for cards to move\n
                - depth: number of cards to move (1,...,12)\n
                - src_height: idx of the top card in `source` pile, so naturally it's a height of this pile -1\n
                - trgt_height: idx of the top card in `target` pile where the cards are to be placed\n

                :optional params:\n

                - facedown_counts: if provided, reveals the next top card in `source` pile if needed.\n
                - seq_depths:      if provided, updates depths of sequences in both piles.
            returns:
                True if `facedown_counts` is provided and a new card was revealed in `source` pile, False otherwise"""
        
        #moving cards:
        tableau_piles[target, trgt_height: trgt_height + depth] = tableau_piles[source, src_height-depth+1: src_height+1]
        tableau_piles[source, src_height-depth+1: src_height+1] = __NONCARD_VALUE
        
        # optional operations:
        res = False
        #reveal facedown card if any:
        if facedown_counts is not None:
            res = SpiderEnv._reveal_topcard(tableau_piles, facedown_counts, source, src_height-depth)

        #updating seqs depths:
        if seq_depths is not None:
            SpiderEnv._update_depths(tableau_piles, seq_depths, target, trgt_height, source, depth)

        return res
        
    def _move_cards_sequence(self, source: int, target: int, depth: int,
                           __N_PILES=N_PILES) -> bool:
        """Does not check the validity of a sequence to move (this should be ensured by calling `max_sequences_depths()`).
            Only checks  whether it's possible to move this sequence on the desired position (by checking whether the top card of `source` is of a higher by +1 rank and there is enough space to store a sequence).
            
            :params:
                - source: idx of a pile from which cards are taken\n
                - target: idx of a pile - destination for a sequence of cards to move\n
                - depth: number of cards to move\n
            :returns:
                True if the move was successful. _In this case, _facedown_tableu_cards is updated if needed (decreased by 1 if a new card is revealed in `source` pile).
                If `target==source` returns False
                """
        assert 0<=target <__N_PILES and 0<=source <__N_PILES
        tableau_piles, stock_pile, counts, seq_depths = self._unpack_state(self._internal_state)
        
        if target !=source and 0<depth<=seq_depths[source]:
            assert depth<= SpiderSpace._get_max_sequences_depths(tableau_piles[source]), f"Incorrect depth ({depth}) of a sequence to move {tableau_piles[source]}"
            
            idx_top_cards = SpiderSpace._get_top_cards_indices(tableau_piles)
            
            if source not in idx_top_cards[0]: #source is empty
                #raise Exception(f"Trying to move a sequence from an empty pile {source}")
                return False
            
            #source pile is not empty
            src_height  = idx_top_cards[1][idx_top_cards[0]==source][0] # idx of a top card in Source pile
            
            if tableau_piles[source, src_height-depth+1]<=0:
                return False
                #raise Exception(f"Trying to move a sequence with facedown cards from pile {source}")
                
            elif target not in idx_top_cards[0]: #if target is a free pile
                self._facedown_tableu_cards -= SpiderEnv._swap_cards(tableau_piles, source, target, depth, src_height, 0, 
                                                                    facedown_counts=counts,  seq_depths=seq_depths)
                return True
                     
            elif depth<=src_height+1: #check idx topcard is >depth 
                trgt_height = idx_top_cards[1][idx_top_cards[0]==target][0]+1 #+1 so that an empty space above is chosen
            
                if (trgt_height + depth< self.observation_space.pile_size   and 1 < SpiderSpace.get_rank(tableau_piles[target, trgt_height-1])==SpiderSpace.get_rank(tableau_piles[source, src_height-depth+1]) +1 ):
                    self._facedown_tableu_cards -= SpiderEnv._swap_cards(tableau_piles, source, target, depth, src_height, trgt_height, 
                                                                        facedown_counts=counts, seq_depths=seq_depths)
                    # print(self._internal_state)
                    return True
                else:
                    # not possible to move a seq of cards because of max pile size limitations
                    # and the target card is not appropriate
                    return False 
            else:
                return False
                #raise Exception(f"Trying to move more cards ({depth}) than there is in the pile {source}")
        else:
            return False

    
        
    def _deal_cards(self, __NONCARD_VALUE=NONCARD_VALUE, __N_PILES=N_PILES, __HIDE_VALUE=SpiderSpace.HIDE_VALUE, __DTYPE=SpiderSpace.DTYPE) -> bool:
        """This method checks, whether there are cards in the stock pile and all the tableau piles have at least one card, and if so deals cards from stock pile.
            Once cards are dealt, updates sequences depths."""
        tableau_piles, stock_pile, counts, seq_depths = self._unpack_state(self._internal_state)

        #check whhether there are cards in stock pile and each tableu pile is not empty (depth>=1)
        if (n_cards_stock:=counts[-1]) and np.all(seq_depths>0):
            # new cards positions = count cards +1 -1 (because of indexing)
            # ALTERNATIVELY, `SpiderSpace._get_top_cards_indices(tableau_piles)` can be used here
            new_cards_positions = np.count_nonzero(tableau_piles - __NONCARD_VALUE, axis=1) # num of cards (both face-up and face-down) in each pile
            # (N_PILES, )
            
            if np.any(new_cards_positions>self.observation_space.pile_size):
                print(f"Impossible to deal cards, max size of cards reached. Rearrange cards on the table and then try to deal cards again")
                return False
            elif np.count_nonzero(tableau_piles[i_piles:=np.arange(tableau_piles.shape[0],dtype=__DTYPE),
                                  new_cards_positions] - __NONCARD_VALUE)==0:
                # moving 10 cards from stock_pile to tableau_piles:
                #assigning using *advanced indexing*
                tableau_piles[i_piles,new_cards_positions] = stock_pile[n_cards_stock - __N_PILES: n_cards_stock] - __HIDE_VALUE
                stock_pile[n_cards_stock- __N_PILES:n_cards_stock]  = 0
                counts[-1] -= __N_PILES

                #updating sequences depths:
                suit_mask = SpiderSpace.get_suit(tableau_piles[i_piles,new_cards_positions-1])== SpiderSpace.get_suit(tableau_piles[i_piles,new_cards_positions])
                rank_mask = SpiderSpace.get_rank(tableau_piles[i_piles,new_cards_positions-1])== 1+ SpiderSpace.get_rank(tableau_piles[i_piles,new_cards_positions])
                d_mask = rank_mask & suit_mask
                seq_depths[d_mask ] += 1
                seq_depths[(~ d_mask) | (new_cards_positions==0) ] = 1 # if a pile was empty or seq broken

                
                self._stock_len = counts[-1]
                return True
            else:
                raise Exception(f"Cards are found on places where zeros expected: {tableau_piles}")
        else:
            False # it was not possible to deal another set of cards bc one of piles (stock or tableu) is empty


    def _discard_complete_sequences(self,
        __NONCARD_VALUE=NONCARD_VALUE, __COMPLETE_SEQ_LEN = SpiderSpace.HIGHEST_RANK, __N_TARGETS=N_TARGETS,
        __DTYPE=SpiderSpace.DTYPE) -> int:
        """Call this method once an action (move/deal cards) was performed to check whether a complete sequence was built and if so, discard it from the pile.
            Returns number of completed sequences (or 0).
        """
        tableau_piles, stock_pile, facedown_counts, seq_depths = self._unpack_state(self._internal_state)
        
        mask = seq_depths==__COMPLETE_SEQ_LEN
        n_completed_seqs =  np.sum(mask)
        if n_completed_seqs:
            assert 0<=self._complete_sequences<=__N_TARGETS
            self._complete_sequences += n_completed_seqs
        
            heights = np.count_nonzero(tableau_piles[mask] - __NONCARD_VALUE, axis=1) #heights 
            i_piles = np.arange(tableau_piles.shape[0], dtype=__DTYPE)[mask]

            for i_pile, height in zip(i_piles, heights):
                assert len(seq:=tableau_piles[i_pile, height - __COMPLETE_SEQ_LEN: height]) ==__COMPLETE_SEQ_LEN and np.all(np.diff(seq)==-1) and np.all((suits:=SpiderSpace.get_suit(seq))==suits[0] ), f"Error: assembled sequence {seq} does not satisfy homogenity and ascendence properties"
                assert seq_depths[i_pile]==__COMPLETE_SEQ_LEN, "Incorrect depth info, check internal state integrity"
                
                #updating piles:
                tableau_piles[i_pile, height - __COMPLETE_SEQ_LEN: height] = __NONCARD_VALUE
                # reveal topcards where necessary:
                SpiderEnv._reveal_topcard(tableau_piles, facedown_counts, i_pile, height - __COMPLETE_SEQ_LEN -1)
            
                #updating seqs depths:
                seq_depths[i_pile] = SpiderSpace._get_max_sequences_depths(tableau_piles[i_pile])
            
            return n_completed_seqs
        else:#no complete seqs were assembled
            return 0
        
    #
    ##############################################
    # REWARDS functions, which heuristically reflect how beneficial an action is in the current game state
    @staticmethod
    def _deal_cards_reward(n_stock_cards:int, facedown_counts, depths, coeff = -1.0,
                          alpha:float=1.0, beta:float=1.0, gamma: float=1.0,
                          __N_INIT_STOCK_CARDS = SpiderSpace.N_STOCK_CARDS, __N_INIT_FACEDOWN_CARDS= SpiderSpace.N_FACEDOWN_CARDS,
                          __HIGHEST_RANK=SpiderSpace.HIGHEST_RANK, __N_PILES=SpiderSpace.N_PILES
        ) -> float:
        """Dealing cards all the time without considering formed sequences and number of facedown cards is irresponsible. 
        That's why these factors must be taken into consideration. 
        There maybe reverse linear dependence between these factors and how favourable dealing cards is.
        This function takes number of cards in the stock (`n_stock_cards` with coef `alpha`), number of facedown cards on the tableau (`facedown_counts` with coef `beta`) and depths of the top cards sequences (`depths` with coef `gamma`). 
        The result is a product of these factors, shifted by +1: `coeff*(1+ alpha*...)*(1+beta*...)*(1+gamma...)`.
        """
        if coeff!=0:
            # the reward is supposed to be negative
            coef_left_stock_cards = (__N_INIT_STOCK_CARDS - n_stock_cards + __N_PILES)/__N_INIT_STOCK_CARDS
            coef_left_facedown_cards = sum(facedown_counts)/ (__N_INIT_FACEDOWN_CARDS)
            coef_seq_sizes = sum(__HIGHEST_RANK - depths)/(__N_PILES*__HIGHEST_RANK)
            return (coeff * 
                    (1+alpha*coef_left_stock_cards)*
                    (1+ beta*coef_left_facedown_cards) * 
                    (1+ gamma*coef_seq_sizes))
        else:
            return 0
    # @staticmethod
    # def _extend_sequence_reward(n_stock_cards:int, depths, 
    #                             coeff = DEFAULT_REWARDS["complete sequence"],
    #                             gamma: float=(DEFAULT_REWARDS["complete sequence"]-DEFAULT_REWARDS["extend sequence"])/DEFAULT_REWARDS["complete sequence"],
    #                             __RANKS_RANGE_=SpiderSpace.HIGHEST_RANK-SpiderSpace.LOWEST_CARD) -> float:
    #     """Assigns small reward for completing initial segments and big rewards for completing big sequences. 
    #     :params:
    #         depths: resulting depths of sequences
    #         gamma: geometric progression coefficient, choose one <1
    #     """
    #     if coeff!=0:
    #         return coeff* np.power(gamma, __RANKS_RANGE_ -depths[depths>1]).sum()
    #     else:
    #         return 0
    #
    ##################################
    
    @staticmethod
    @njit(inline='always')
    def _unpack_state(state: npt.NDArray[np.integer]| list[list[int]], 
        __N_PILES=N_PILES, __N_STOCK_CARDS=SpiderSpace.N_STOCK_CARDS,
        __I_STOCK_PILE=SpiderSpace.I_STOCK_PILE, __I_FACEDOWN_CNTS=SpiderSpace.I_FACEDOWN_CNTS, __I_DEPTHS_CNTS=SpiderSpace.I_DEPTHS_CNTS
        ) -> tuple[npt.NDArray[np.integer]]:
        """Returns tuple (tableau piles, stock pile, facedown counts, depths)"""
        return state[:__N_PILES], state[__I_STOCK_PILE, : __N_STOCK_CARDS], state[__I_FACEDOWN_CNTS, :__N_PILES+1], state[__I_DEPTHS_CNTS, :__N_PILES]

    
    # def check_state(self) -> bool:
    #     """check correctnbess of counts"""
    #     tableau_piles, stock_pile, counts = self._unpack_state(self._internal_state)
    #     return np.all(counts == np.count_nonzero(self._internal_state[:SpiderEnv+1], axis=1))
    
    def render(self,  fancy_mode:bool=True, 
               print_auxillary_info: bool=False, show_agents_obs: bool=False, 
               __HIDE_VALUE=SpiderSpace.HIDE_VALUE):
        """"Renders observed state. Example of output:
                Stock cards: 50
                🂠 🂠 🂠 🂠 🂠 🂠 🂠 🂠 🂠 🂠
                🂠 🂠 🂠 🂠 🂠 🂠 🂠 🂠 🂠 🂠
                🂠 🂠 🂠 🂠 🂠 🂠 🂠 🂠 🂠 🂠
                🂠 🂠 🂠 🂠 🂠 🂠 🂠 🂠 🂠 🂠
                🂠 🂠 🂠 🂠 🂳 🃂 🂩 🃅 🂮 🃙
                🃋 🂲 🂴 🃖 
            If fancy_mode is turned off, more readable output is produced.
        """
        piles,stock,counts,depths = SpiderEnv._unpack_state(self._internal_state)
        print(f"Number of complete sequences: {self._complete_sequences}\nStock cards: {counts[-1]}")
        i_nzpiles, idx_nz = np.nonzero(piles)# * (piles>0))
        n_render_rows_limit = max(idx_nz)+1#self.observation_space.shape[1]

        f_card2str = SpiderSpace._card_to_fancy_str if fancy_mode else SpiderSpace._card_to_str
        print('\n'.join(" ".join(f_card2str(piles[i_pile, i_cards_row]) for i_pile in range(len(piles)))  
                            for i_cards_row in range(n_render_rows_limit)))

        if print_auxillary_info:
            print(f"Face-down cards counts: {counts}")
            print(f"Depths of top sequences: {depths}")
            print("Cards in stock: "+" ".join(f_card2str(card - __HIDE_VALUE) for card in stock[:counts[-1]]))

        if show_agents_obs:
            print(f"Agent's observations: {SpiderEnv._unpack_state(self.get_obs())}")
    # implement if any external resources are used by env, like an open window when rendering
    # def close(self):
    #     if self.window is not None:
    #         pygame.display.quit()
    #         pygame.quit()
            

    def user_play(self, render_mode:str='fancy', 
                  _show_agents_obs: bool=False,
                  __N_TARGETS=SpiderSpace.N_TARGETS,
        inactivity_timeout: int|None=20) -> bool:
        """Test function. Type 'exit' to escape. 
            Returns True if all the N_TARGETS sequences were completed (victory), False if 'exit'"""
        d_actions = {'move': self._move_cards_sequence, 'deal': self._deal_cards}

        while 1:
            print(f"\n{inactivity_timeout} secs to make a move, otherwise game stops\n")
            if (inp:=timed_input(timeout=inactivity_timeout, prompt='Input action: ')) is None:
                break
            else:
                l_action = inp.lower().split()
    
            res= False
            if len(l_action):
                if l_action[0]=='exit':
                    return False
                                
                s_func = l_action[0]
                
                if s_func=='move':
                    args = [int(a) for a in l_action[1:]]
                    if len(args)==2:
                        args.append(1)
                    res = d_actions[s_func](*args)
                else:
                    res = d_actions[s_func]()

                #check whether any sequence has been completed:
                self._discard_complete_sequences()
                
                if self.get_game_status():
                    print("\nVictory! You completed all the {__N_TARGETS} sequences! Congratulations!")
                    return True
                
            if not len(l_action) or res:
                clear_output()
                self.render(fancy_mode=(render_mode=='fancy'), 
                         print_auxillary_info=True,
                           show_agents_obs=_show_agents_obs)
    def random_play(self, render_mode:str='fancy', 
        n_max_iterations: int=50000,
                    verbosity: int=1,
                    _render_state_timeout:int=0, _diagnostic_mode:bool=False,
                    _sample_valid_actions_only: bool=True, 
                    _show_agents_obs: bool=False,
                    __N_TARGETS=SpiderSpace.N_TARGETS,
                    inactivity_timeout: int|None=60) -> bool:
        """Test function.
            Returns True if all the N_TARGETS sequences were completed (victory), False otherwise. """
        cum_rew = 0

        action_sampler = self._sample_valid_action if _sample_valid_actions_only else  self.action_space.sample
        
        for game_iter in range(n_max_iterations):
            a = action_sampler()
            obs, reward, terminated, truncated, info = self.step(a)
        
            cum_rew+=reward

            #diagnostic info
            if verbosity>1:
                if _render_state_timeout and  game_iter %_render_state_timeout==0:
                    if not _diagnostic_mode:
                        clear_output()
                    if verbosity>2:
                        print('action ', self._unflatten_move_action(a)if  0<=a<self.action_space.n else a)
                        print('reward',  reward)
                        #print('terminated', terminated)
                    self.render(fancy_mode=(render_mode=='fancy'), 
                    print_auxillary_info=True,
                    show_agents_obs=_show_agents_obs)
        
            if terminated or truncated:
                 break

        if not _diagnostic_mode:
            clear_output()
        if verbosity>=1:
            print("Avg reward per"+ (' valid' if _sample_valid_actions_only else '') +f" action: {cum_rew/game_iter}")

        self.render(fancy_mode=(render_mode=='fancy'), 
                    print_auxillary_info=True,
                    show_agents_obs=_show_agents_obs)
        
        if self.get_game_status():
            if verbosity>=1:
                print("\nVictory! You completed all the {__N_TARGETS} sequences! Congratulations!")
            return True
            
        return False