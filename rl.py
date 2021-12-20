import gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv
from finta import TA


from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C



import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = ''

#==============================================================================
# CLASS: rl
#==============================================================================
class rl:
    def __init__(self):
        self.root_path = os.getcwd()
        self.log_path = os.path.join(self.root_path, 'logs')
        self.model_path = os.path.join(self.root_path, 'models')
        self.ppo_model_path = os.path.join(self.root_path, 'models', 'ppo_model')
        self.dqn_model_path = os.path.join(self.root_path, 'models', 'dqn_model')
        self.environment_name = 'stocks-v0'


    # READ DATA
    #==============================================================================
    def read_data(self) -> pd.DataFrame:
        df = pd.read_csv('data/gme.csv')
        
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', ascending=True, inplace=True)
        df.set_index('Date', inplace=True)

        df['Volume'] = df['Volume'].apply(lambda x: float(x.replace(",", "")))
        df['SMA'] = TA.SMA(df, 12)
        df['RSI'] = TA.RSI(df)
        df['OBV'] = TA.OBV(df)
        df.fillna(0, inplace=True)

        print(df.head())
        return df


    # LOAD ENV
    #==============================================================================
    def load_env(self) -> gym:
        print('\n>>> Loading Environment...')
        env = gym.make(self.environment_name, 
            df = self.read_data(), 
            frame_bound = (5,250), 
            window_size = 5
        )
        #print(env.signal_features)
        #print(env.action_space)
        state = env.reset()
        while True: 
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            if done: 
                print("info", info)
                break
                
        plt.figure(figsize=(15,6))
        plt.cla()
        env.render_all()
        plt.show()
        
        return env


    # ADD SIGNALS
    #==============================================================================
    def add_signals(self):
        env = self.load_env()     
        start = env.frame_bound[0] - env.window_size
        end = env.frame_bound[1]
        prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
        signal_features = env.df.loc[:, ['Low', 'Volume','SMA', 'RSI', 'OBV']].to_numpy()[start:end]
        return prices, signal_features


    # LOAD CUSTOM ENV
    #==============================================================================
    def load_custom_env(self) -> gym:
        
        class custom_env(StocksEnv):       
            _process_data = self.add_signals()

        env2 = custom_env(
            df = self.read_data(), 
            window_size = 12, 
            frame_bound=(12,50)
        )
        print(env2.signal_features)
        print(self.data.head())





