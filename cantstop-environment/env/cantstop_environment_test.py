from cantstop_env import CantstopEnvironment

from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = CantstopEnvironment()
    parallel_api_test(env, num_cycles = 1000)