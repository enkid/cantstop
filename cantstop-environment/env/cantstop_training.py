import random

from cantstop_env import CantstopEnvironment
from cantstop_agent import DumbAgent, RandomAgent

ITERATIONS = 1000

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



for i in range(100):
    current_run_tot = 0
    if (i % 1 == 0):
        print("Iteration: ", i )
    for j in range(ITERATIONS):
        actions = {}
        cont = True
        while cont:    
            if env.activeplayer == 0:
                actions["continue"] = r_agent.get_cont_action(env.temp_board, env.player_boards[env.activeplayer])
                actions["dice"] = r_agent.get_dic_action(env.observations["dice"]["action_mask"])
            else:
                actions["continue"] = r_agent.get_cont_action(env.temp_board, env.player_boards[env.activeplayer])
                actions["dice"] = r_agent.get_dic_action(env.observations["dice"]["action_mask"])
            obs, reward, term, trunc, infos = env.step(actions)
            if env.timestep == 1000:
                cont = False
            if term["continue"] == True:
                current_run_tot += reward["continue"]
                cont = False
        env.reset()
    run_tots += [current_run_tot]
    if current_run_tot > (ITERATIONS * 0.55) and current_run_tot == max(run_tots):
        d_agent.old_weights = d_agent.new_weights.copy()
        print("Found Better weighing")
    #d_agent.mutate()    



best_wins = max(run_tots)
best_run = run_tots.index(best_wins)

print("Best Run : ", best_run, " Best Wins: ", best_wins)
print("All wins: ", run_tots)
print("Highest Value move: ", d_agent.old_weights.index(max(d_agent.old_weights)))
    
#with open("Action_priority.flat", 'w') as file:
#    for i in d_agent.old_weights:
#        file.write(f"{i}\n" )
