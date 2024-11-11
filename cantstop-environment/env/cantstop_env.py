from pettingzoo import ParallelEnv

from cantstop_mask import CantStopActionMask

from gymnasium.spaces import Discrete, MultiDiscrete, Tuple

from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector, wrappers

import random

import functools


#                 2 3 4 5 6  7  8  9 10 11 12   
board_shape = [0, 3,5,7,9,11,13,11,9,7, 5, 3]

roll_perm = [[0,1,2,3], [0, 2, 1, 3], [0,3,2,1]]


class CantstopEnvironment(ParallelEnv): #need to figure out how to integrate multiple agents - not sure if AEC or parallel works better
    metadata = {
        "name": "cantstop_environment_v0",
    }

    def __init__(self):
        self.activeplayer = None
        self.player_boards = None
        self.temp = None
        self.temp_board = None
        self.viable_choices = None
        self.possible_agents = ["dice", "continue"]
        self.timestep = None
        

    def reset(self, seed=None, options=None):
        self.activeplayer = 0
        self.player_boards = [[-1]*12] + [[-1]*12]
        self.temp_board = [-1]*12
        self.viable_choices = list(range(2,13))
        self.timestep = 0

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}


        observation = {
            a: (
                self.activeplayer,
                self.player_boards,
                self.temp_board,
            )
            for a in self.agents
        }

        dicerolls = [ random.randint(1,6) for _ in range(4)] #get the dice rolls

        action_mask = CantStopActionMask(dicerolls, self.temp_board)
        #print("action mask line 63:", action_mask)

        self.observations = {
            "dice" : {"observation": observation, "action_mask": action_mask.mask},
            "continue" : {"observation": observation, "action_mask": [1]*2},
        }

        return self.observations, self.infos

    
    def __str__(self):
        board_str = "2 3 4 5 6 7 8 9 101112\n"
        for v in range(1, max(board_shape)+1): #skip first item, which should always be zero anyways.
            line = ""
            for h in range(1, len(board_shape)):
                if board_shape[h] >= v:
                    if self.temp_board[h] == v:
                        line = line + "y"                 
                    elif self.player_boards[0][h] == v and self.player_boards[1][h] == v:
                        line = line+"x"
                    elif self.player_boards[1][h] == v:
                        line = line+"1"
                    elif self.player_boards[0][h] == v:
                        line = line + "0"
                    else:
                        line = line + "|"
                else:
                    line = line + " "
                line = line + " "
            board_str = board_str + line + "\n"
        return board_str


    def win_check(self):
            win_count = 0

            #print(self.player_boards[self.activeplayer])
            for i in range(1, len(self.player_boards[self.activeplayer])):
                #print(self.player_boards[self.activeplayer][i])
                if self.player_boards[self.activeplayer][i] == board_shape[i]:
                    #print("Position : ", i, " player value:", self.player_boards[self.activeplayer][i]+1, "board_shape ", board_shape[i])
                    win_count = win_count + 1
            if win_count >= 3:
                print("Got to win on step: ", self.timestep)
                print(self)
                return True
            else:
                return False



    def step(self, actions):
        dice_action = actions["dice"] #should be a number in the form (total of 2 dice) * 12 + (total of 2 dice), where a total of 0 means dice not selected. 0 means no possible move
        continue_action = actions["continue"] #action will be a continue or stop
        win=False


        if dice_action == 0 : #no possible move
            #print("AI picked no possible move")
            self.temp_board = [-1]*12
            #switch players / add negative reward?

        else:
            if dice_action == 132: #setting special case due to way modulo works, probably should find a better ways
                dice_tot2 = 11
                dice_tot1 = 11
            else:
                dice_tot2 =int( dice_action %12)
                dice_tot1 =int( dice_action / 12)
            #print("Making move: ", dice_tot1, " ", dice_tot2, " on temp_board: ", self.temp_board)
            #print("temporary board levels: dice tot 1 : ", dice_tot1, " ", self.temp_board[dice_tot1], " dice tot 2 : ", dice_tot2, " ", self.temp_board[dice_tot2] )

            if dice_tot1 != 0 and self.temp_board[dice_tot1] < board_shape[dice_tot1]:
                if self.temp_board[dice_tot1] == -1: #add peg to temporary board
                    self.temp_board[dice_tot1] = self.player_boards[self.activeplayer][dice_tot1]
                if self.temp_board[dice_tot1] < board_shape[dice_tot1]:
                    self.temp_board[dice_tot1] = self.temp_board[dice_tot1] + 1 #advance peg on temporary board


            if dice_tot2 != 0 and self.temp_board[dice_tot2] < board_shape[dice_tot2]:
                if self.temp_board[dice_tot2] == -1: #add peg to temporary board
                    self.temp_board[dice_tot2] = self.player_boards[self.activeplayer][dice_tot2]
                if self.temp_board[dice_tot2] < board_shape[dice_tot2]:
                    self.temp_board[dice_tot2] = self.temp_board[dice_tot2] + 1 #advance peg on temporary board

            
            if continue_action == 0: #agent says stop

                for i in range(1, len( self.temp_board)): #make temporary advancement permament
                    if self.temp_board[i] != -1:
                        self.player_boards[self.activeplayer][i] = self.temp_board[i]
                
                #print("AI says stop, new player board: ", self.player_boards[self.activeplayer])
               
                self.temp_board = [-1]*12

                win = self.win_check()
                #self.activeplayer = abs(self.activeplayer - 1) let's just use one active player for now

            
        

        self.terminations = {a: False for a in self.agents}
        self.rewards = {a:-0.01 for a in self.agents} #trying to win as fast as possible. This sohould be a good baseline, but eventually we will want to make it adversarial

        if win:
                self.rewards = {a:1 for a in self.agents}
                self.terminations = {a: True for a in self.agents}

        #Should I have a truncation?
        self.truncations = {a: False for a in self.agents}
        self.timestep += 1


        observation = {
            a: (
                self.activeplayer,
                self.player_boards,
                self.temp_board,
            )
            for a in self.possible_agents
        }

        
        dicerolls = [ random.randint(1,6) for _ in range(4)] #get the dice rolls

        action_mask = CantStopActionMask(dicerolls, self.temp_board)

        self.observations = {
            "dice" : {"observation": observation, "action_mask": action_mask.mask},
            "continue" : {"observation": observation, "action_mask": [1]*2},
        }

        #dummy infos
        self.infos = {a: {} for a in self.agents}

        if any(self.terminations.values()) or all(self.truncations.values()):
            self.agents = []

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos
        

    def render(self):
        print(self)


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return   Tuple((Discrete(2), MultiDiscrete([12]*2), Discrete(12))) #MultiDiscrete([2, [12, 12], [12]]) - it didn't like this #active player, player_boards, temp_board
      
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if agent == "dice":
            return Discrete (144)
        if agent == "continue":
            return Discrete (2)