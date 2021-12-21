import tensorflow as tf 

import gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv
from finta import TA


from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C



import os
import datetime as dt
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
        self.stock = 'gme'


    # READ DATA
    #==============================================================================
    def read_data(self) -> pd.DataFrame:
        df = pd.read_csv(f'data/{self.stock}.csv')
        
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
        print(f"[{dt.datetime.now().strftime('%H:%M:%S')}]: Loading {self.stock} Default Environment...\n")
        env = gym.make(self.environment_name, 
            df = self.read_data(), 
            frame_bound = (5,250), 
            window_size = 5
        )
        print(env.signal_features)
        print(env.action_space)
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

        print(f"[{dt.datetime.now().strftime('%H:%M:%S')}]: Environment Loaded.")
        return env


    # LOAD CUSTOM ENV
    #==============================================================================
    def load_custom_env(self) -> gym:
        data = self.read_data()
        env = gym.make(self.environment_name, 
            df = data, 
            frame_bound = (5,250), 
            window_size = 5
        )

        def add_signals(env):
            start = env.frame_bound[0] - env.window_size
            end = env.frame_bound[1]
            prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
            signal_features = env.df.loc[:, ['Low', 'Volume','SMA', 'RSI', 'OBV']].to_numpy()[start:end]
            return prices, signal_features

        class custom_env(StocksEnv):       
            _process_data = add_signals

        env2 = custom_env(
            df = data, 
            window_size = 12, 
            frame_bound=(12,50)
        )
        print(env2.signal_features)
        print(data.head())

        state = env2.reset()
        while True: 
            action = env2.action_space.sample()
            n_state, reward, done, info = env2.step(action)
            if done: 
                print("info", info)
                break

     
        plt.figure(figsize=(15,6))
        plt.cla()
        env2.render_all()
        plt.show()

        return env2



    # TRAIN MODEL
    #==============================================================================
    def train_model(self) -> A2C:     
        dt_start = dt.datetime.now()
        env = self.load_custom_env()
        env_maker = lambda: env
        env = DummyVecEnv([env_maker])
        model = A2C('MlpLstmPolicy', env, verbose=1) 
        print(model.learn(total_timesteps=1000000))
        dt_end = dt.datetime.now()
        print(f"[{dt_start.strftime('%H:%M:%S')}]: Training {self.stock} Model...")
        print(f"[{dt_end.strftime('%H:%M:%S')}]: {self.stock} Model Trained.")
        print(f'{dt_start - dt_end}')
        return model


    # EVAL MODEL
    #==============================================================================
    def evaluate_model(self) -> A2C:
        dt_start = dt.datetime.now()
        model = self.train_model()
        dt_end = dt.datetime.now()
        print(f"[{dt_start.strftime('%H:%M:%S')}]: Training {self.stock} Model...")
        print(f"[{dt_end.strftime('%H:%M:%S')}]: {self.stock} Model Trained.")
        print(f'[Time Taken]:{ dt_start - dt_end}')

        dt_start = f"[{dt.datetime.now().strftime('%H:%M:%S')}]: Loading Custom Environment..."
        env = self.load_custom_env()
        dt_end = f"[{dt.datetime.now().strftime('%H:%M:%S')}]: Environment Loaded."
        print(f'{dt_start}\n{dt_end}')

        dt_start = f"[{dt.datetime.now().strftime('%H:%M:%S')}]: Evaluating {self.stock} Model..."
        obs = env.reset()
        while True: 
            obs = obs[np.newaxis, ...]
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if done:
                print("info", info)
                break
        
        print(f"[{dt.datetime.now().strftime('%H:%M:%S')}]: Training {self.stock} Model...\n")




        def check_gpu(self):
            if tf.test.gpu_device_name(): 
                print(f'Default GPU Device:{tf.test.gpu_device_name()}')
            else:
                print("Please install GPU version of TF")