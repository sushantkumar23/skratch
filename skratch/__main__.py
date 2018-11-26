
from agents.vpg_agent import VPGAgent
from trading.run_experiment import Runner

import gym
import gym_trading

if __name__ == "__main__":

    env = gym.make("fxtrading-v0")

    runner = Runner(VPGAgent, env)
    runner.run_experiment()
