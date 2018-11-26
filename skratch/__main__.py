
from agents.vpg_agent import VPGAgent
from trading.run_experiment import Runner

import gym
from gym_trading.envs import TradeEnv

if __name__ == "__main__":

    symbols = ['AUDUSD']
    start_date = '2012-01-01'
    end_date = '2017-12-31'

    for symbol in symbols:
        env = TradeEnv(symbol=symbol, start_date=start_date, end_date=end_date)
        env.get_data(symbol)

        runner = Runner(VPGAgent, env)
        runner.run_experiment()
