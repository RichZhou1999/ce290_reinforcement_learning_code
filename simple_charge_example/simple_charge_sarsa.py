import numpy as np
import gym
from  simple_charge_env import Simple_charge_env
from simple_charge_env import max_current,current_interval, start_time_max, step, battery_ah
from pathlib import Path
import random

# Parameters
epsilon = 0.9                           # greedy for exploration
total_episodes = 10000                  # episodes to learn
max_steps = 144                         # time steps
alpha = 0.05                            # learning rate
gamma = 0.95                            # discount factor

env = Simple_charge_env()
reset_state = {
    'current_soc': 0.2159713063120908,
    'target_soc': 0.6871365930818221,
    't0': 0,
    'tf': 144
}

class SARSA_agent:
    def __init__(self, env, alpha=0.05, gamma=0.95, epsilon=0.9) -> None:
        
        self.env = env
        self.Q = np.zeros((env.observation_space.shape[0], env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.I_list = []

    def choose_action(self, state):
        
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # exploration
        else:
            return np.argmax(self.Q[state, :])  # exploitation
    
    def update_Q(self, state, action, reward, next_state, next_action):
        
        target = reward + self.gamma * self.Q[next_state, next_action]
        error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * error

    def train(self, num_episodes, mode=None, reset_state=None):
        
        for i in range(num_episodes):
            
            if mode == 'reset_with_values':
                try:
                    state = env.reset_with_values(
                        reset_state['start_soc'],
                        reset_state['end_soc'],
                        reset_state['t0'],
                        reset_state['tf']
                    )
                except:
                    state = env.reset_with_values(
                        0.2159713063120908,
                        0.6871365930818221,
                        0,
                        144)
            else:
                state = self.env.reset()

            start_soc, _, _, _, _, _ = state
            action = self.choose_action(state)
            self.I_list.append(env.current_list[action])
            done = False

            while not done:
                next_state, reward, done, tru, info = self.env.step(action)
                next_action = self.choose_action(next_state)
                self.update_Q(state, action, reward, next_state, next_action)

                current_soc, target_soc, current_time, end_time, I_max, price = next_state

                if done and i % 5 == 0:
                    print(f"====== EP: {i} | Reward: {round(reward, 2)} =======")
                    print("start_soc: ", round(start_soc, 3))
                    print("current_soc: ", round(current_soc, 3))
                    print("target_soc: ", round(target_soc, 3))
                    print("current_time: ", current_time)
                    print("end_time: ", end_time)
                    print("I:", self.I_list)
                    print()

                state = next_state
                action = next_action

    
    def test(self, num_episodes):
        
        rewards = []
        for i in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = np.argmax(self.Q[state, :])
                next_state, reward, done, tru, info = self.env.step(action)
                state = next_state
                episode_reward += reward

            rewards.append(episode_reward)

        return np.mean(rewards)


simple_agent = SARSA_agent(env)
simple_agent.train(num_episodes=2000)
