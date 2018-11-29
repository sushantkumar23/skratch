# __main__.py

import datetime

from agents.vpg_agent import VPGAgent
from trading.run_experiment import Runner
from gym_trading.envs import TradeEnv

if __name__ == "__main__":

    symbols = ['AUDUSD', 'EURUSD']
    start_date = '2012-01-01'
    end_date = '2017-12-31'

    for symbol in symbols:
        env = TradeEnv(symbol=symbol, start_date=start_date, end_date=end_date)

        current_datetime = datetime.datetime.now()
        base_dir = "./summaries/{}-{}".format(
            symbol,
            current_datetime.strftime("%Y-%m-%dT%H:%M:%S"))

        runner = Runner(VPGAgent, env, base_dir=base_dir)
        runner.run_experiment()
