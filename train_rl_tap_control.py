import gym
import numpy as np
from stable_baselines3 import PPO
from gym import spaces

class LTCEnv(gym.Env):
    def __init__(self):
        super(LTCEnv, self).__init__()
        self.observation_space = spaces.Box(low=np.array([0.8, 0]), high=np.array([1.2, 2000]), dtype=np.float32)
        self.action_space = spaces.Discrete(33)
        self.reset()
    def step(self, action):
        tap_change = action - 16
        self.tap_position += tap_change
        self.tap_position = np.clip(self.tap_position, -16, 16)
        base_voltage = 1.0 + self.tap_position * 0.00625
        voltage_drop = (self.load / 2000.0) * 0.05
        self.voltage = base_voltage - voltage_drop + np.random.normal(0, 0.003)
        self.load += np.random.uniform(-50, 50)
        self.load = np.clip(self.load, 100, 2000)
        voltage_penalty = abs(self.voltage - 1.0)
        overload_penalty = max(0, self.load - 1500) / 500.0
        tap_penalty = abs(tap_change) * 0.01
        reward = -(voltage_penalty + overload_penalty + tap_penalty)
        obs = np.array([self.voltage, self.load], dtype=np.float32)
        done = False
        return obs, reward, done, {}
    def reset(self):
        self.voltage = np.random.uniform(0.95, 1.05)
        self.load = np.random.uniform(500, 1500)
        self.tap_position = 0
        return np.array([self.voltage, self.load], dtype=np.float32)

if __name__ == "__main__":
    env = LTCEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("models/ppo_tap_controller") 