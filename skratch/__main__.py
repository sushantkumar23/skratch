
from agents.vpg_agent import VPGAgent
from trading.run_experiment import Runner

import gym
import gym_trading

if __name__ == "__main__":

    def create_agent_fn(sess):
        agent = VPGAgent(sess)
        return agent

    env = gym.make("fxtrading-v0")

    runner = Runner(create_agent_fn, env)
    runner.run_experiment()
