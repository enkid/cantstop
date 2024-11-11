import random


import numpy as np


from cantstop_env import CantstopEnvironment


#let's start with a very simple algorithm that trains the y/n decision to continue. Goal will be to minimize time it takes to win.  We'll make the taking dice decision random at first

class ContinueAgent:
    
    def __init__(self):
        cont_array = []
        for i in range(100):
            cont_array += random.random(15, 25)
        self.old_weights = cont_array
        self.new_weights = cont_array

    def mutate(self):
        for i in range(len(self.old_weights)):
            self.new_weights = self.old_weights + random.random(-5,5)
        
    def update_weights(self):
        self.old_weights = self.new_weights


    def calc_dice_gains(self, temp_board, player_board):
        total_dice_gains = 0
        for i in range(len(temp_board)):
            if temp_board != -1:
                total_dice_gains += temp_board[i] - player_board[i]
        return total_dice_gains

