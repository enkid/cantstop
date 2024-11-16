import random


import numpy as np


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
    
    def get_cont_action(self, env):
        temp_board = env.temp_board
        player_board = env.player_boards[env.activeplayer]
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


#class that picks random thing to do out of possible actions
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
    
    def get_cont_action(self, env):
        return random.randint(0, 1)


#helper functions for NN

def sigmoid (x):
    return 1.0 / (1.0 + np.exp(-x)) #sigmoid squashing

        


class NNAgent():
    H = 50 #hidden node size
    D = 3 * 12 + 3  #Dimensionality
    decay_rate = 0.9995
    learning_rate = 1e-5

    def __init__(self, previous_weights=None):
        self.obs_record = []
        self.hs_record = []
        self.dlogps = []
        self.rew_record = []
        if previous_weights == None:
            self.model = {}
            self.model["W1"] = np.random.randn(self.H,self.D) / np.sqrt(self.D) #"Xavier" Initialization - TODO, understand what this is
            self.model["W2"] = np.random.randn(self.H) / np.sqrt(self.H)
        else:
            self.model = previous_weights

    def calc_dice_gains(self, temp_board, player_board):
        total_dice_gains = 0
        for i in range(len(temp_board)):
            if temp_board[i] != -1:
                total_dice_gains += temp_board[i] - player_board[i]
        return total_dice_gains


    def win_check(self, env):
            win_count = 0

            for i in range(1, len(env.player_boards[env.activeplayer])):
                if env.player_boards[env.activeplayer][i] == env.board_shape()[i] or env.temp_board[i] == env.board_shape()[i]:
                    win_count = win_count + 1
            if win_count >= 3:
                return True
            else:
                return False


    def preprocess_obs(self,env):
        #create array of all observations in float
        dice_gains = self.calc_dice_gains(env.temp_board, env.player_boards[env.activeplayer]) #seems like something the AI should care about, so trying to make it obvious
        win_check = self.win_check(env)
        tmp = np.concatenate( (env.observations["continue"]["observation"], np.array([dice_gains]), np.array([win_check])))
        prepocessed_obs = tmp.astype(float).ravel()
        return prepocessed_obs

    def get_actions(self, env):
        actions = {}
        actions["dice"] = self.get_dic_action(env)
        actions["continue"] = self.get_cont_action(env)


    def get_dic_action(self, env):
        action_mask = env.observations["continue"]["action_mask"]
        return random.choice( [i for i, x in enumerate(action_mask) if x == 1])
    
    def get_cont_action(self, env):
        preproc_obs = self.preprocess_obs(env)
        aprob, h = self.policy_forward(preproc_obs)
        action = 1 if np.random.uniform() < aprob else 0
        
        
        self.record_step_observation(action, aprob, preproc_obs, h) #this needs to move to get_action
        return action
    
    def policy_forward(self, preproc_obs):
        h = np.dot(self.model['W1'], preproc_obs)
        h[h<0] = 0 # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = sigmoid(logp)
        return p, h # return probability of continuing, and hidden state

    def record_step_observation(self, action, action_prob, obs, hidden):
        self.obs_record.append(obs)#this needs to be the previous observation
        self.hs_record.append(hidden) #this needs to be the previous state
        self.dlogps.append(action - action_prob) #need to check to see if this is backwards

    def policy_backward(self):
        ep_obs = np.vstack(self.obs_record)
        ep_hs = np.vstack(self.hs_record)
        ep_dlogps = np.vstack(self.dlogps)
        ep_reward = np.vstack(self.reward) #assuming reward is just 1 or zero...
        ep_dlogps *= ep_reward
        dW2 = np.dot(ep_hs.T, ep_dlogps).ravel()
        dh = np.outer(ep_dlogps, self.model['W2'])
        dh[ep_hs <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh.T, ep_obs)
        return {'W1':dW1, 'W2':dW2}


    
    def record_game(self, reward):
        self.reward = [reward] * len(self.obs_record) # reward is uniform right as no way to differentiate most impactful actions at this point, I think
        grad = self.policy_backward()

        rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.items() } # rmsprop memory

        grad_buffer = { k : np.zeros_like(v) for k,v in self.model.items() }

        for k in self.model: grad_buffer[k] += grad[k] # accumulate grad over batch
        for k,v in self.model.items():
            
            g = grad_buffer[k] # gradient
            
            rmsprop_cache[k] = self.decay_rate * rmsprop_cache[k] + (1 - self.decay_rate) * g**2
            self.model[k] += self.learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        self.obs_record = []
        self.hs_record = []
        self.dlogps = []
        self.rew_record = []