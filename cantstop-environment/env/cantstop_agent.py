import random


import numpy as np


from cantstop_env import CantstopEnvironment


#let's start with a very simple algorithm that trains the y/n decision to continue. Goal will be to minimize time it takes to win.  We'll make the taking dice decision random at first

class DumbAgent: #Just uses a rank order of possible moves and a set time to commit after getting dice results
                #In other words, it does not take board state into account. Next agent will be some sort of neural network that takes into account the board states
    
    

    def __init__(self, preset_weights=None):
        self.stop_gains = 3

        if preset_weights != None:
            self.old_weights = preset_weights
            self.new_weights = self.old_weights.copy()
        else:
            self.old_weights = [143-i for i in range(144)]
            self.new_weights = self.old_weights.copy() 
            for i in range(0,100):
                self.mutate() #starting with a random configuration
        
        
        
    def update_weights(self):
        self.old_weights = self.new_weights

    def mutate(self):
        self.new_weights = self.old_weights.copy()
        for i in range(10):
            i1 = random.randint(0,143)
            i2 = random.randint(0,143)
            while i1 == i2:
                i2 = random.randint(0,143)
            self.new_weights[i2] = self.old_weights[i1]
            self.new_weights[i1] = self.old_weights[i2]


    def calc_dice_gains(self, temp_board, player_board):
        total_dice_gains = 0
        for i in range(len(temp_board)):
            if temp_board[i] != -1:
                total_dice_gains += temp_board[i] - player_board[i]
        #print(temp_board, " ", player_board)
        #print('total_dice_gains: ', total_dice_gains)
        return total_dice_gains
    
    def get_cont_action(self, temp_board, player_board):
        if self.calc_dice_gains(temp_board, player_board) > self.stop_gains:
            #print("Keeping Gains ")
            return 0
        else:
            #print("Risking It")
            return 1
        
    def get_dic_action(self, action_mask):
        options = [i for i, x in enumerate(action_mask) if x==1]
        options_weights = [self.new_weights[i] for i in options]
        return self.new_weights.index(max(options_weights))

class RandomAgent():
    def __init__(self):
        pass

    def get_actions(self, action_mask):
        actions = {}
        actions["dice"] = self.get_dic_action(action_mask)
        actions["continue"]  = self.get_cont_action()
        return actions

    def get_dic_action(self, action_mask):
        return random.choice( [i for i, x in enumerate(action_mask) if x == 1])
    
    def get_cont_action(self):
        return random.randint(0, 1)


