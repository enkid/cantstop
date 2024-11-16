import random

from cantstop_env import CantstopEnvironment
from cantstop_agent import DumbAgent, RandomAgent, NNAgent


EPISODES = 200000

env = CantstopEnvironment()
obs, infos = env.reset()



"""Load weights for the DumbAgent"""
#weights = None
#with open("Action_priority.flat", 'r') as file: #This is the with the best weights after many runs against a random agent
#    weights = [int(line.strip()) for line in file]

#d_agent = DumbAgent(weights)
#test_agent = DumbAgent(weights)




run_tots = []

r_agent = RandomAgent()

d_agent = DumbAgent()

nn_agent = NNAgent()


run_tots = []

print("Starting training loop")

for i in range(EPISODES):
    actions = {}
    cont = True
    while cont:    #each game is an iteration of the loop
        if env.activeplayer == 0:
            actions["continue"] = nn_agent.get_cont_action(env)
            actions["dice"] = r_agent.get_dic_action(env.observations["dice"]["action_mask"])
        else:
            actions["continue"] =d_agent.get_cont_action(env)
            actions["dice"] = r_agent.get_dic_action(env.observations["dice"]["action_mask"])
        obs, reward, term, trunc, infos = env.step(actions)
        if env.timestep == 1000:
            cont = False
        if term["continue"] == True:
            #if reward["continue"] == 1:
                #print('Win')
                #print(env)
            cont = False
    nn_agent.record_game(reward["continue"])
    env.reset()
    run_tots += [1 if reward["continue"] == 1 else 0]
    DISPLAY_ITERS = 1000
    if((i+1) % DISPLAY_ITERS == 0):
        print("Episode: ", i )
        s = sum(run_tots)
        print(f"{s} wins in last {DISPLAY_ITERS}" )
        run_tots = []



#step through the final result
env.reset()

cont = True
while cont:    #each game is an iteration of the loop
    input(env)

    if env.activeplayer == 0:
        actions["continue"] = nn_agent.get_cont_action(env)
        print("Neural Agent chooses: ", actions["continue"])
        actions["dice"] = r_agent.get_dic_action(env.observations["dice"]["action_mask"])
        print("Random Agent chooses : ", actions["dice"] % 12, " ", actions["dice"] / 12)

    else:
        actions["continue"] =d_agent.get_cont_action(env)
        print("Dumb Agent chooses: ", actions["continue"])
        actions["dice"] = d_agent.get_dic_action(env.observations["dice"]["action_mask"])
    obs, reward, term, trunc, infos = env.step(actions)
    print("Temp Board", env.temp_board)
    if env.timestep == 1000:
        cont = False
    if term["continue"] == True:
        cont = False

print(env)



#with open("Action_priority.flat", 'w') as file:
#    for i in d_agent.old_weights:
#        file.write(f"{i}\n" )
