import numpy as np
import numpy.typing as npt
from numba import njit

import gymnasium as gym
from gymnasium import spaces

class SpiderSpace(gym.spaces.Box):
    NONCARD_VALUE = 0
    LOWEST_CARD = 1
    HIGHEST_RANK = 13
    HIGHEST_CARD = 4*HIGHEST_RANK
    HIDE_VALUE = - 2*(HIGHEST_CARD - LOWEST_CARD +1)
    
    N_PILES = 10 # +1 for undealt cards
    N_CARDS = 2*(HIGHEST_CARD - LOWEST_CARD + 1)
    N_STOCK_CARDS =  (N_CARDS// N_PILES//2) * N_PILES #50
    N_FACEDOWN_CARDS = N_CARDS - N_STOCK_CARDS - N_PILES#init number of facedown cards in the game, 44
    N_TARGETS = N_CARDS//(HIGHEST_RANK -LOWEST_CARD+1) #number of sequences to complete (A,2,...,10,J,Q,K)x8

    I_STOCK_PILE = N_PILES 
    I_FACEDOWN_CNTS = N_PILES +1
    I_DEPTHS_CNTS = N_PILES +2

    DTYPE = np.int8

    SUITS = {0:'♠', 1:'♥', 2:'♣', 3: '♦'}
    RANKS = {**{i: str(i) for i in range(2, 11)}, **{11: 'J', 12: 'Q', 13: 'K', 1: 'A'}}
    FANCY_CARDS = {   0: {1: '🂡', 2: '🂢', 3: '🂣', 4: '🂤', 5: '🂥', 6: '🂦', 7: '🂧', 8: '🂨', 9: '🂩', 10: '🂪', 11: '🂫', 12: '🂭', 13: '🂮'},
                      1: {1: '🂱', 2: '🂲', 3: '🂳', 4: '🂴', 5: '🂵', 6: '🂶', 7: '🂷', 8: '🂸', 9: '🂹', 10: '🂺', 11: '🂻', 12: '🂽', 13: '🂾'},
                      2: {1: '🃑', 2: '🃒', 3: '🃓', 4: '🃔', 5: '🃕', 6: '🃖', 7: '🃗', 8: '🃘', 9: '🃙', 10: '🃚', 11: '🃛', 12: '🃝', 13: '🃞'},
                      3: {1: '🃁', 2: '🃂', 3: '🃃', 4: '🃄', 5: '🃅', 6: '🃆', 7: '🃇', 8: '🃈', 9: '🃉', 10: '🃊', 11: '🃋', 12: '🃍', 13: '🃎'}
                  }
    
    def __init__(self, n_suits: int=4, counts_features:bool=True, maxsize: int = 64, 
                 vectorize: bool=False,
                 dtype=np.int8, seed: int|None=None):
        """
            :params:
                n_suits: number of suits in the game. The default value is 4 (Spades,...)
                counts_features: if True, the last 2 rows (features) of the space represent counts in the tableau piles + stock piles. In the beginning of a game counts are 5,5,5,5,4,4,4,...,4,50 for facedown carts and 1,...1 for max depths feature. 
                maxsize: max number of cards in one pile. Should not be lesser than N_STOCK_CARDS (50)
                vectorize: bool (False by default) - if True, adds a dimension to observation space spatial signature (N_PILES+..., maxsize) -> (N_PILES+..., maxsize, 4), which would yield more comprehensible for deep learning agents results. All the cards, represented internally as 1<=i<=52 cards will be represented as [0,..,i%13, 0] 4-vectors 
                dtype: preferred numpy dtatatype, set to int8 for efficiency
                seed: dummy seed param
            returns:
                np.array of shape (N_PILES+1+counts_feature, maxsize) of dtype
                """
        assert 1<=n_suits<=4
        assert maxsize>= SpiderSpace.N_STOCK_CARDS, "make sure maxsize is more than expected number of undealt cards (e.g. 50)"
        
        self.pile_size =maxsize
        self.n_suits = n_suits
        self.counts_features = counts_features

        if not vectorize or n_suits==1:
            shape = (SpiderSpace.N_PILES + 1 + 2*int(self.counts_features), maxsize)
        else:
            shape = (SpiderSpace.N_PILES + 1 + 2*int(self.counts_features), maxsize, n_suits)
        # if dtype==np.int8:
        low = min(SpiderSpace.LOWEST_CARD,SpiderSpace.NONCARD_VALUE)  + SpiderSpace.HIDE_VALUE
        high = n_suits * SpiderSpace.HIGHEST_RANK
                
        super().__init__(low=low,  high=high,  shape=shape,  dtype = dtype,  seed=seed)
        # else:
        #     low = np.full(shape, min(SpiderSpace.LOWEST_CARD,SpiderSpace.NONCARD_VALUE)  + SpiderSpace.HIDE_VALUE, dtype = dtype)
        #     high =np.full(shape, SpiderSpace.HIGHEST_CARD, dtype = dtype)
            
        #     super().__init__(low=low,  high=high,   dtype = dtype, seed=seed)
        
        #self.low, self.high = SpiderSpace.LOWEST_CARD + SpiderSpace.HIDE_VALUE, SpiderSpace.HIGHEST_CARD
    
    def sample(self, hide_cards: bool=True,
               __LOWEST_CARD=LOWEST_CARD, __HIGHEST_RANK=HIGHEST_RANK, __HIGHEST_CARD=HIGHEST_CARD, __NONCARD_VALUE=NONCARD_VALUE, __N_CARDS=N_CARDS, __CARDS_RANGE=-LOWEST_CARD+HIGHEST_RANK+1,
               __N_STOCK_CARDS=N_STOCK_CARDS, __N_PILES=N_PILES, __HIDE_VALUE=HIDE_VALUE,  
               __I_STOCK_PILE=I_STOCK_PILE, __I_FACEDOWN_CNTS=I_FACEDOWN_CNTS, __I_DEPTHS_CNTS=I_DEPTHS_CNTS,
               __DTYPE=DTYPE) -> npt.NDArray[np.integer]:
        """Generates a valid Spider preset in the form [n_piles, max_cards_per_pile].
            :params:
                hide_cards: if True, the cards in the stock pile (the last row) and the cards other than last in the tableau piles are considered as face-down, i.e. invisible to an agent. This is done by subtracting the HIGHEST_CARD.
                facedown_counts: if True, the last row[:N_PILES] in the returned numpy array is reserved to store counts of facedown cards in the tableau piles and stock pile (50). In the beginning of a game the counts are expected to be as follows: 6,6,6,6,5,5,5...,5, 50
            :returns:
                np.array of small (int8) integers"""
        #cards are represented by integers
        cards = np.arange(__LOWEST_CARD, self.n_suits*__HIGHEST_RANK +1, dtype=__DTYPE)
        cards = np.hstack([cards] * (__N_CARDS//(self.n_suits* __CARDS_RANGE)))
        np.random.shuffle(cards)
        empty_space = np.full(self.shape[:2], __NONCARD_VALUE, dtype=__DTYPE)

        n_undealt_cards = __N_STOCK_CARDS
        n_dealt_cards = len(cards) - n_undealt_cards
        
        #cards are dealt to piles sequentially, e.g. 6,6,6,6,5,5,5,5,5,5:
        min_cards_to_pile = n_dealt_cards//__N_PILES
        n_total_dealt_cards = 0
        for i in range(__N_PILES):
            n_cards_to_pile = min_cards_to_pile + (i< n_dealt_cards% __N_PILES)
            empty_space[i, :n_cards_to_pile ] += cards[n_total_dealt_cards: (n_total_dealt_cards := n_total_dealt_cards+ n_cards_to_pile )]

            #subtracting 104 if hide_cards:
            if hide_cards:
                empty_space[i, :n_cards_to_pile -1] += __HIDE_VALUE
            # count facedown cards in the last row if necessary
            if self.counts_features:
                empty_space[__I_FACEDOWN_CNTS, i] = n_cards_to_pile-1
                empty_space[__I_DEPTHS_CNTS, i] = 1
                
            
        # remaining cards in the draw pile (e.g. 50)
        empty_space[__N_PILES, :n_undealt_cards] = cards[-n_undealt_cards:] - 2*__HIGHEST_CARD if hide_cards else 0
        if self.counts_features:
            #assert self.pile_size==self.shape[1]>__N_STOCK_CARDS, "Make sure there is space for counts: the last row in the matrix should be reserved"
            empty_space[__I_FACEDOWN_CNTS, __N_PILES] = __N_STOCK_CARDS
                    
        return empty_space

    
    @staticmethod
    @njit(inline='always')
    def get_rank(cards, 
                 __LOWEST_CARD=LOWEST_CARD, __HIGHEST_RANK=HIGHEST_RANK,
                __RANKS_RANGE = HIGHEST_RANK  - LOWEST_CARD+1) -> int:
        #always positive
        return (cards - __LOWEST_CARD) % __RANKS_RANGE + __LOWEST_CARD 
    @staticmethod
    @njit(inline='always')
    def get_suit(cards, n_suits: int=4, 
                 __LOWEST_CARD=LOWEST_CARD, __HIGHEST_RANK=HIGHEST_RANK,
                __RANKS_RANGE = HIGHEST_RANK  - LOWEST_CARD+1) -> int:
        return  ((cards - __LOWEST_CARD) % (n_suits* __RANKS_RANGE)) // __HIGHEST_RANK

    @staticmethod
    @njit(inline='always')
    def is_sequential(moving_card, destination_card,  
                 __LOWEST_CARD=LOWEST_CARD, __HIGHEST_RANK=HIGHEST_RANK,
                __RANKS_RANGE = HIGHEST_RANK  - LOWEST_CARD+1) -> bool:
        """Checks whether it'spossible to move a `moving_card` onto a `destination_card`"""
        return (moving_card % __RANKS_RANGE >0) & ((destination_card - moving_card) % __RANKS_RANGE ==1)

        
    @staticmethod
    # @njit(inline='always')
    def _card_to_str(card: int, ranks: dict|list=RANKS, suits:dict|list=SUITS) -> str:
        """card: int. If card<0, ' 🂠' is returned, if 0, then '  ' is returned"""
        return f"{ranks[SpiderSpace.get_rank(card)]}{suits[SpiderSpace.get_suit(card)]}" if card>0 else ' 🂠' if card<0 else '  '
    @staticmethod
    def _card_to_fancy_str(card: int, fancy_cards: dict[dict[int, str]]=FANCY_CARDS) -> str:
        """card: int. If card<0, ' 🂠' is returned, if 0, then ' ' is returned"""
        return fancy_cards[SpiderSpace.get_suit(card)][SpiderSpace.get_rank(card)] if card>0 else '🂠' if card<0 else ' '
        
    @staticmethod
    def _vectorize(state: npt.NDArray[np.integer], n_suits: int=4, _counts_features: bool=True,
                 __NONCARD_VALUE=NONCARD_VALUE, __N_PILES=N_PILES,
                 __I_STOCK_PILE=I_STOCK_PILE, __I_FACEDOWN_CNTS=I_FACEDOWN_CNTS, __I_DEPTHS_CNTS=I_DEPTHS_CNTS
                 ) -> npt.NDArray[np.integer]:
        """Each card number in the space/state is translated into a 4-vector, e.g. Ace of Spades -> [1,0,0,0], King of Hearts -> [0,13,0,0]. The output has the same number of dimensions as the input +1.
        This function does not handle states with the last row reserved for facedown cards counts. Use `_counts_features=True` to recalculate these counts for "vectorized" cards.
        The function is called when vectorized cards are requested by `SpiderEnv` constructor (the argument `vectorize_obs` passed to `SpiderSpace` constructor) in order to represent agent's observations by a 3D tensor.
            :params:
                state: numpy array 
                n_suits: number of suits in the game (1,2,3,4)
                _counts_features: if True, the last but one row-tensor is reserved to count face-down cards (negative numbers if any), and the last one stores max depths of top cards sequences per each pile
                                Should be put False if tableau piles are input as state
        """
        nonzero_idc = np.nonzero(state - __NONCARD_VALUE) # 2-tuple (rows_idc, cols_idc)
        ##ranks and suits of nonzero cards (1D arrays)
        nonzero_cards = state[nonzero_idc] # np.copy(state[nonzero_idc])
        #nonzero_cards[nonzero_cards<0] -= SpiderSpace.HIDE_VALUE 
        suits_idc = SpiderSpace.get_suit(nonzero_cards, n_suits) # i.e. Spades -> 0, ...
        ranks = SpiderSpace.get_rank(nonzero_cards)  * np.sign(nonzero_cards)
        
        empty_space = np.full((*state.shape, n_suits), __NONCARD_VALUE, dtype=state.dtype) 
        empty_space[*nonzero_idc, suits_idc] = ranks

        if _counts_features: 
            #the last but one row for face-down cards counts (also stock pile, i.e. N_PILES+1 counts)
            empty_space[__I_FACEDOWN_CNTS, :__N_PILES+1] = SpiderSpace._get_facedown_counts(empty_space[:__I_STOCK_PILE+1])# np.count_nonzero(empty_space[:, :-1], axis=1)
            empty_space[__I_DEPTHS_CNTS, :__N_PILES]   = SpiderSpace._get_max_sequences_depths(empty_space[:__N_PILES])
            
        return empty_space
        
    def _devectorize() -> npt.NDArray[np.integer]:
        ...

    @staticmethod
    def _pad_matrix(matrix, arithmetic_axis:int=1, 
                    __NONCARD_VALUE=NONCARD_VALUE):
        return np.pad(matrix,  
                      [(0,0)]*arithmetic_axis + [(0,1)] + [(0,0)]*(matrix.ndim-arithmetic_axis-1), constant_values=__NONCARD_VALUE)
    @staticmethod
    def _get_top_cards_indices(tableau_piles: npt.NDArray[np.integer],
                              _return_difference: bool=False,
                                __NONCARD_VALUE=NONCARD_VALUE) -> npt.NDArray[np.integer]:
        """this method could be useful...
                returns positions of sequences (1 card or elements of ordered sequences) which can be moved to another pile.
                The output is `np.array` of shape `(N_PILES,)`
            params:
                tableau_piles: the first N_PILES of game state (tableau piles)
                _return_difference: if True, the function returns the tuple `(cards_indices, differences)`, where `differences` represents shifted differences between cards per each pile. The latter is used by `_get_max_sequences_depths(tableau_piles)` function.
                Example:
                >>> SpiderSpace._get_top_cards_indices(np.array([[1,4,3,15,14,13,12,0,0],
                            [1,4,3,16,15,13,12,0,0],
                              [1,4,3,2,16,15,14,0,0],
                              [3,2,1,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0],
                              [2,3,1,0,0,0,0,0,0],
                              [16,15,14,0,0,0,0,0,0],
                              [1,4,9,9,5,4,3,-2,9]]))
                (array([0, 1, 2, 3, 5, 6, 7]), array([6, 6, 6, 2, 2, 2, 8]))
        """
        #determining arithmetic axis
        ar_axis = int(len(tableau_piles.shape)>1) #0 if tableau_piles is 1D array, 1 otherwise
        #difference between cards per each pile (to detect ascending sequences (-1))
        padded_tableau = SpiderSpace._pad_matrix(tableau_piles, ar_axis) 
        d = np.diff(padded_tableau, axis=ar_axis)# will be truncated
        idc_topcards = np.nonzero(d - __NONCARD_VALUE)
    
        mask = np.append(np.diff(idc_topcards[0]), 1).astype(bool) 
        bnd_coords = tuple(bnd_ax[mask] for bnd_ax in idc_topcards)
        return bnd_coords if not _return_difference else (bnd_coords, d)#[:,:-1])
        
    @staticmethod
    def _get_max_sequences_depths(tableau_piles: npt.NDArray[np.integer],
                                    __HIGHEST_CARD = HIGHEST_CARD, __NONCARD_VALUE=NONCARD_VALUE, __HIDE_VALUE=HIDE_VALUE,
                                    __DTYPE=DTYPE
                                    ) -> npt.NDArray[np.integer]:
        """Accepts a valid game state (tableau piles). Supports also a vactorized form of tableau_piles (in this case returns 3D tensor with depths corresponding to a top suit cards)
            Instance for 3D case:
                >>> SpiderSpace._get_max_sequences_depths(SpiderSpace._vectorize(np.array([
                        [1,4,3,15,14,13,12,0,0],
                        [1,4,3,16,15,13,12,0,0],
                          [1,4,3,2,16,15,14,0,0],
                          [3,2,1,0,0,0,0,0,0],
                          [2,3,1,0,0,0,0,0,0],
                          [16,15,14,0,0,0,0,0,0],
                          [1,4,9,9,5,4,3,-2,9]]), _counts_features=False))
                array( [[2, 0, 0, 0],
                        [2, 0, 0, 0],
                        [0, 3, 0, 0],
                        [3, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 3, 0, 0],
                        [1, 0, 0, 0]])
            Returns an array each position of which is max number of cards in the sequence it is possible yo move to another pile. Bounded by 1 and 12.
        """
        ar_axis = int(len(tableau_piles.shape)>1) #0 if tableau_piles is 1D array, 1 otherwise
        #difference between cards per each pile (to detect ascending sequences (-1))
        #print(tableau_piles)
        d = np.diff(tableau_piles, axis=ar_axis)# will be truncated
        idc_topcards = np.nonzero(d - __NONCARD_VALUE)

        if ar_axis:
            #print(d[:,:10,:])
            mask = np.append(np.diff(idc_topcards[0]), 1).astype(bool) 
            bnd_coords = tuple(bnd_ax[mask] for bnd_ax in idc_topcards)
            
            # print(bnd_coords)
            #selecting zero diffs before bnds (when cards are of same rank) and hiding them:
            #cols_mask = (np.arange(d.shape[1], dtype=__DTYPE) < bnd_coords[1][:,None]) & np.all(d==0, axis=ar_axis+1)# (d.shape[0],d.shape[1])
            
            # print(d[:2])
            if len(tableau_piles.shape)>2:
                #cols_mask = (np.arange(d.shape[1], dtype=__DTYPE) < bnd_coords[1][:,None])[...,None] & (d[bnd_coords[0]]==__NONCARD_VALUE)
                cols_mask = (np.arange(d.shape[1], dtype=__DTYPE) < bnd_coords[1][:,None]) & np.all(d[bnd_coords[0]]==__NONCARD_VALUE, axis=-1)
                
                r, c, *_rest_dim = np.nonzero(cols_mask)
            else: 
                # print(bnd_coords)
                # print(np.arange(d.shape[1], dtype=__DTYPE) < bnd_coords[1][:,None])
                # print(d)
                cols_mask = (np.arange(d.shape[1], dtype=__DTYPE) < bnd_coords[1][:,None]) & (d[bnd_coords[0]]==__NONCARD_VALUE)
                r, c = np.nonzero(cols_mask)
                
            # print(np.squeeze(np.all(d==0, axis=1)))
            #print(cols_mask)
            
            d[bnd_coords[0][r], 
                c, 
                tuple(bnd_coords[i][r] for i in range(2,len(tableau_piles.shape))) if len(tableau_piles.shape)>2 else None
            ] = __HIDE_VALUE
            #print("final indices which vals to be assigned to HIDE_VALUE : ", bnd_coords[0][r], *c)
        
        else: #for 1D arrays
            if len(idc_topcards[0])>0:
                bnd_coords = idc_topcards[0][-1]
                
                #selecting zero diffs before bnds (when cards are of same rank) and hiding them:
                cols_mask = (np.arange(d.shape[0])<bnd_coords) & (d==0)# (d.shape[1],)
                d[cols_mask] = __HIDE_VALUE
            else:
                return 0 # the array (pile) is completely empty (0-filled)
        #suits difference
        sd = np.diff(SpiderSpace.get_suit(tableau_piles), axis=ar_axis).astype(bool) # True if two cards are of different suit
        d[sd & (d==-1)] -= __HIGHEST_CARD #subtracts 52 from -1 at the edge of a valid sequence diff([3,2,1,0])=[-1,-1,-1], so that -1 -1 -1 -> -1 -1 -10 and will yield depth=3 
        # bnd_rows, bnd_cols, ... contain coords of right non-zeros (bounds)
        # assigning right bounds of every pile with -1 if the next card in a pile is greater by one
        d[bnd_coords] = np.where(d[bnd_coords] < __NONCARD_VALUE, -1, +1)
        
        d[np.nonzero(d==-1)] = __NONCARD_VALUE
        #print(d)
        consequtive_zeros = (np.flip(d, axis=ar_axis)==__NONCARD_VALUE).cumprod(axis=ar_axis).sum(axis=ar_axis)
        # depths = n consequitive zeros - n of zeros in the tableu_piles
        depths = consequtive_zeros- (np.flip(tableau_piles, axis=ar_axis)==__NONCARD_VALUE).cumprod(axis=ar_axis).sum(axis=ar_axis)+1 # subtracting num of zeros in tableau_piles
        return depths

    @staticmethod
    @njit(inline='always')
    def _get_facedown_counts(piles: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        """piles: tableau piles and stock pile state matrix. 
            Returns number of face-down cards"""
        return np.count_nonzero(piles*(piles<0), axis=1)

        
    # @staticmethod
    # def validate(state: npt.NDArray[np.integer], facedown_cards: bool=True, facedown_counts: bool=True) -> bool:
    #     """Validate the state/space. 
    #         :params:
    #             facedown_cards: if True, negative values are included in range checks
    #             #facedown_counts: if True and facedown_cards, the last row is checked to contain number of negative values in every column"""
    #     state_orig = state
    #     state=np.array(state)
    #     if facedown_counts: 
    #         state[-1] = 0
    #     if facedown_cards:
    #         state[state<0] -= SpiderSpace.HIDE_VALUE    

    #     #print(state)
    #     nonzero_idc = np.nonzero(state - SpiderSpace.NONCARD_VALUE)
    #     nonzero_cards = state[nonzero_idc]
    #     #print(len(nonzero_cards), len(set(nonzero_cards)))
    #     nonzero_cards = nonzero_cards.flatten()
    #     unique_cards = np.unique(nonzero_cards)
    #     #print(np.count_nonzero(state, axis=1))
    #     #print(state_orig*(state_orig<0))
    #     return (len(nonzero_cards) == 2*len(unique_cards) <= 2*SpiderSpace.N_CARDS
    #             and np.all((SpiderSpace.LOWEST_CARD <=unique_cards) & (unique_cards <=SpiderSpace.HIGHEST_CARD)) 
    #             and ((np.all((cnts:=SpiderSpace.calculate_facedown_counts(state_orig[:-1]))==state_orig[-1, :len(cnts)]) if facedown_cards and facedown_counts else True))
    #            )

    @staticmethod
    #@njit(inline='always')
    # works well with 2D but higher dimensions are supported poorly
    def _left_align_rows(state: npt.NDArray[np.integer], __NONCARD_VALUE=NONCARD_VALUE) -> npt.NDArray[np.integer]:
        """"shifts rows (piles) with zeros left. For example, [[0,1,0,2], [-9,0,8,0]] -> [[1,2,0,0],[9,0,-8,0]]
            Do not apply this transformation for the last two rows (stock pile and facedown counts)"""
        # if len(state.shape)>2 :
        #     print(np.argsort(np.any(state!=__NONCARD_VALUE, axis=2)[:,None]))
        # return np.take_along_axis(state, 
        #                           np.argsort(np.any(state!=__NONCARD_VALUE, axis=1)[:,None] if len(state.shape)>2 else state, axis=1),
        #                           axis=1)#(((state[:,0]!=__NONCARD_VALUE) | (np.count_nonzero(state, axis=1)==0))), axis=1)
        if len(state.shape)>2 :
            zero_mask = ~np.any(state, axis=-1)
        elif len(state.shape)==2:
            zero_mask = ~state.astype(bool)
        else:
            raise NotImplementedError("Not implemented for {len(state.shape)} numpy-arrays")
        shift = np.argmax(~zero_mask, axis=1)
        return state[np.arange(state.shape[0])[:, None], 
                        (np.arange(state.shape[1]) + shift[:,None]) % state.shape[1] ]
        