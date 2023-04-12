#
# import numpy as np
#
# import gym
# from gym import logger, spaces
#
#
#
# '''
#     observation_space:
#                         high        low
#         current_soc:     1           0
#         target_soc:      1           0
#         start_time:     1440         0
#         end_time:       2880         0
#         current_time:   end_time   start_time
#             P:         10000(kw)     0
#           I_max:        1e5          0
#     action_space: 0 - 100 amp ( with the interval of 0.5A)
# '''
#
# # TIME PARAMETERS
# step = 10
# total_time = 1440
# start_time_max = total_time / step
# end_time_max = total_time / step * 2
# min_charge_intervals = 60 / step
#
# # BATTERY PARAMETERS
# max_current = 25                                # A
# min_current = 0                                 # A
# current_interval = 1                            # A
# voltage = 400                                   # V
# battery_kwh = 60                                # kWh
# battery_ah = battery_kwh * 1e3 / voltage        # Ah
# resistance = 0                                  # Ohm
# power_boundary = 10 * 1e3                       # kW
# power_boundary_decrease_point = 0.8
#
# # PRICE PARAMETERS
# emission_max_value = 1                          # $/kWh
#
# class Simple_charge_env:
#     def __init__(self):
#
#         self.current_soc = np.random.uniform(0, 0.8)
#         self.target_soc = np.random.uniform(self.current_soc + 0.1, 1)
#
#         self.start_time = np.random.randint(0, start_time_max)
#         self.end_time = np.random.randint(start_time_max + self.start_time, start_time_max + self.start_time +  min_charge_intervals)
#
#         self.action_space = spaces.Discrete(int((max_current-min_current)/current_interval))
#
#         # self.observation_space = spaces.Box(-np.array([0, 0, 0, 0, 0, 0, 0]),
#         #                                     np.array([1, 1, 1e5, 1e5, 1e5, 1e5, 1e5]), dtype=np.float32)
#
#         self.observation_space = spaces.Box(-np.array([0, 0, 0, 0,0, 0]),
#                                             np.array([ 1e5, 1e5, 1e5, 1e5,1e5,1e5]), dtype=np.float32)
#
#
#         # self.voltage = 0.4
#
#         self.current_list = np.linspace(0, max_current, int((max_current-min_current)/current_interval))
#
#         self.resistance = float(resistance)
#
#         self.power_boundary = float(power_boundary)
#
#         self.power_boundary_decrease_point = float(power_boundary_decrease_point)
#
#         self.current_time = self.start_time
#
#         self.charge_interval = step
#
#         self.battery_ah = float(battery_ah)
#         self.voltage = voltage
#
#
#     def get_power_limit(self):
#         if self.current_soc < self.power_boundary_decrease_point:
#             return self.power_boundary
#         else:
#             return self.power_boundary - (self.power_boundary)/ (1.0-self.power_boundary_decrease_point)*(self.current_soc-self.power_boundary_decrease_point)
#
#     def get_voltage(self):
#         return self.voltage
#
#     def get_I_limit(self):
#         # I = (-U + sqrt(U**2+8*R*P))/(4R)
#         if self.resistance == 0:
#             return self.current_power_limit/self.voltage
#         return (-self.voltage + np.sqrt(self.voltage**2 + 8 * self.resistance * self.current_power_limit))/(4*self.resistance)
#
#     def get_per_kwh_price(self):
#         """get the dynamic electricity price at current time"""
#         t = self.current_time % start_time_max
#         return emission_max_value / ((start_time_max/2)**2) * (t - (start_time_max/2))**2
#
#     def get_reward(self, a):
#         time = self.current_time % start_time_max
#         I_max = self.I_max
#         x = np.linspace(0, int(start_time_max), int(start_time_max+1))
#         y = emission_max_value/((start_time_max/2)**2) * (x-(start_time_max/2))**2
#
#         # max_y = y[0]
#         current_list = np.linspace(0, max_current, int(max_current / current_interval) + 1)
#
#         current = min(I_max, current_list[a])
#         # reward = (max_y - y[int(time)])/max_y * current * step / 60
#         price = self.get_per_kwh_price()
#         # print(price == y[int(time)])
#         reward = -price * (current + 2 * current * resistance / voltage) * step / 60
#         # reward = -y[int(time)] * (current + 2 * current * resistance / voltage) * step / 60
#
#         return reward
#
#
#     # def get_reward(self, a):
#     #     I_limit = self.get_I_limit()
#     #     price = self.get_per_kwh_price()
#     #     current_list = np.linspace(0, max_current, int(max_current / current_interval) + 1)
#     #
#     #     current = min(I_limit, current_list[a])
#     #     reward = -price * (current + 2 * current * resistance / voltage) * step / 60
#     #
#     #     return reward
#
#     def reset(self):
#         self.current_soc = np.random.uniform(0, 0.8)
#         self.target_soc = np.random.uniform(self.current_soc + 0.1, 1)
#
#         self.start_time = np.random.randint(0, start_time_max)
#         self.end_time = np.random.randint(start_time_max + self.start_time, start_time_max + self.start_time +  min_charge_intervals)
#         self.current_time = self.start_time
#
#         self.current_power_limit = self.get_power_limit()
#         self.voltage = self.get_voltage()
#         self.I_max = self.get_I_limit()
#         self.price = self.get_per_kwh_price()
#
#
#         return np.array([self.current_soc,
#                          self.target_soc,
#                          self.current_time,
#                          self.end_time,
#                          self.I_max,
#                          self.price
#                          ])
#
#
#         # return np.array([self.current_soc,
#         #                  self.target_soc,
#         #                  self.start_time,
#         #                  self.end_time,
#         #                  self.current_time,
#         #                  self.current_power_limit,
#         #                  self.I_max
#         #                  ])
#
#     def reset_with_values(self,
#                           current_soc,
#                           target_soc,
#                           start_time,
#                           end_time,):
#         self.current_soc = current_soc
#         self.target_soc = target_soc
#
#         self.start_time = start_time
#         self.end_time = end_time
#         self.current_time = self.start_time
#
#         self.current_power_limit = self.get_power_limit()
#         self.voltage = self.get_voltage()
#         self.I_max = self.get_I_limit()
#         self.price = self.get_per_kwh_price()
#
#
#         return np.array([self.current_soc,
#                          self.target_soc,
#                          self.current_time,
#                          self.end_time,
#                          self.I_max,
#                          self.price
#                          ])
#
#
#         # return np.array([self.current_soc,
#         #                  self.target_soc,
#         #                  self.start_time,
#         #                  self.end_time,
#         #                  self.current_time,
#         #                  self.current_power_limit,
#         #                  self.I_max
#         #                  ])
#
#     def step(self, action):
#         current = self.current_list[action]
#         if current > self.I_max:
#             current = self.I_max
#         self.current_soc += self.charge_interval * current / self.battery_ah / 60
#         self.current_time += 1
#         self.current_power_limit = self.get_power_limit()
#         self.voltage = self.get_voltage()
#         self.I_max = self.get_I_limit()
#         self.price = self.get_per_kwh_price()
#
#         observation = np.array([self.current_soc,
#                          self.target_soc,
#                          self.current_time,
#                          self.end_time,
#                          self.I_max,
#                          self.price
#                          ])
#
#         # observation = np.array([self.current_soc,
#         #                          self.target_soc,
#         #                          self.start_time,
#         #                          self.end_time,
#         #                          self.current_time,
#         #                          self.current_power_limit,
#         #                          self.I_max
#         #                          ])
#         reward = self.get_reward(action)
#         terminated = False
#         if ((self.current_soc >=  self.target_soc) or (self.end_time == self.current_time)):
#
#             terminated = True
#             if self.current_soc < self.target_soc:
#                     reward += -999
#                     # reward += -abs(self.target_soc - self.current_soc) * emission_max_value * battery_ah *(1 + max_current * resistance/voltage)
#
#
#         info = {}
#
#         return observation, reward, terminated, False, info
#
#
#
import numpy as np

import gym
from gym import logger, spaces

'''
    observation_space: 
                        high        low
        current_soc:     1           0
        target_soc:      1           0
        start_time:     1440         0
        end_time:       2880         0
        current_time:   end_time   start_time
            P:         10000(kw)     0
          I_max:        1e5          0
    action_space: 0 - 100 amp ( with the interval of 0.5A)      
'''

# # TIME PARAMETERS
step = 10
total_time = 1440
start_time_max = total_time / step
end_time_max = total_time / step * 2
min_charge_intervals = 60 / step

# BATTERY PARAMETERS
max_current = 25  # A
min_current = 0  # A
current_interval = 1  # A
voltage = 400  # V
battery_kwh = 60  # kWh
battery_ah = battery_kwh * 1e3 / voltage  # Ah
resistance = 0  # Ohm
power_boundary = 10 * 1e3  # kW
power_boundary_decrease_point = 0.8

# PRICE PARAMETERS
emission_max_value = 1  # $/kWh


class Simple_charge_env:
    def __init__(self):

        self.time_step = 10
        self.total_time = 1440
        self.start_time_max = self.total_time / self.time_step
        self.end_time_max = self.total_time / self.time_step * 2
        self.min_charge_intervals = 60 / self.time_step

        # BATTERY PARAMETERS
        self.max_current = 25  # A
        self.min_current = 0  # A
        self.current_interval = 1  # A



        self.current_soc = np.random.uniform(0, 0.8)
        self.target_soc = np.random.uniform(self.current_soc + 0.1, 1)

        self.start_time = np.random.randint(0, self.start_time_max)
        self.end_time = np.random.randint(self.start_time_max + self.start_time,
                                          self.start_time_max + self.start_time + self.min_charge_intervals)

        self.action_space = spaces.Discrete(int((self.max_current - self.min_current) / self.current_interval))

        # self.observation_space = spaces.Box(-np.array([0, 0, 0, 0, 0, 0, 0]),
        #                                     np.array([1, 1, 1e5, 1e5, 1e5, 1e5, 1e5]), dtype=np.float32)

        self.observation_space = spaces.Box(-np.array([0, 0, 0, 0, 0, 0]),
                                            np.array([1e5, 1e5, 1e5, 1e5, 1e5, 1e5]), dtype=np.float32)

        # self.voltage = 0.4

        self.current_list = np.linspace(0, self.max_current, int((self.max_current - self.min_current) / self.current_interval))

        self.resistance = 0

        self.power_boundary_decrease_point = 0.8

        self.power_boundary = 10 * 1e3


        self.current_time = self.start_time

        self.charge_interval = 10

        self.emission_max_value = 1
        self.battery_kwh = 60
        self.voltage = 400
        self.battery_ah = float(self.battery_kwh*1e3/self.voltage)

    def get_power_limit(self):
        if self.current_soc < self.power_boundary_decrease_point:
            return self.power_boundary
        else:
            return self.power_boundary - (self.power_boundary) / (1.0 - self.power_boundary_decrease_point) * (
                        self.current_soc - self.power_boundary_decrease_point)

    def get_voltage(self):
        return self.voltage

    def get_I_limit(self):
        # I = (-U + sqrt(U**2+8*R*P))/(4R)
        if (self.resistance ==0 ):
            return self.current_power_limit/self.voltage
        return (-self.voltage + np.sqrt(self.voltage ** 2 + 8 * self.resistance * self.current_power_limit)) / (
                    4 * self.resistance)

    def get_per_kwh_price(self):
        """get the dynamic electricity price at current time"""
        t = self.current_time % self.start_time_max
        return self.emission_max_value / ((self.start_time_max / 2) ** 2) * (t - (self.start_time_max / 2)) ** 2

    def get_reward(self, a):
        time = self.current_time % self.start_time_max
        I_max = self.I_max
        # x = np.linspace(0, int(self.start_time_max), int(start_time_max + 1))
        # y = emission_max_value / ((start_time_max / 2) ** 2) * (x - (start_time_max / 2)) ** 2

        # max_y = y[0]
        current_list = np.linspace(0, self.max_current, int(self.max_current / self.current_interval) + 1)

        current = min(I_max, current_list[a])
        # reward = (max_y - y[int(time)])/max_y * current * step / 60
        price = self.get_per_kwh_price()
        # print(price == y[int(time)])
        reward = -price * (current + 2 * current * self.resistance / self.voltage) * self.time_step / 60
        # reward = -y[int(time)] * (current + 2 * current * resistance / voltage) * step / 60

        return reward

    # def get_reward(self, a):
    #     I_limit = self.get_I_limit()
    #     price = self.get_per_kwh_price()
    #     current_list = np.linspace(0, max_current, int(max_current / current_interval) + 1)
    #
    #     current = min(I_limit, current_list[a])
    #     reward = -price * (current + 2 * current * resistance / voltage) * step / 60
    #
    #     return reward

    def reset(self):
        self.current_soc = np.random.uniform(0, 0.8)
        self.target_soc = np.random.uniform(self.current_soc + 0.1, 1)

        self.start_time = np.random.randint(0, self.start_time_max)
        self.end_time = np.random.randint(self.start_time_max + self.start_time,
                                          self.start_time_max + self.start_time + self.min_charge_intervals)
        self.current_time = self.start_time

        self.current_power_limit = self.get_power_limit()
        self.voltage = self.get_voltage()
        self.I_max = self.get_I_limit()
        self.price = self.get_per_kwh_price()

        return np.array([self.current_soc,
                         self.target_soc,
                         self.current_time,
                         self.end_time,
                         self.I_max,
                         self.price
                         ])

        # return np.array([self.current_soc,
        #                  self.target_soc,
        #                  self.start_time,
        #                  self.end_time,
        #                  self.current_time,
        #                  self.current_power_limit,
        #                  self.I_max
        #                  ])

    def reset_with_values(self,
                          current_soc,
                          target_soc,
                          start_time,
                          end_time, ):
        self.current_soc = current_soc
        self.target_soc = target_soc

        self.start_time = start_time
        self.end_time = end_time
        self.current_time = self.start_time

        self.current_power_limit = self.get_power_limit()
        self.voltage = self.get_voltage()
        self.I_max = self.get_I_limit()
        self.price = self.get_per_kwh_price()

        return np.array([self.current_soc,
                         self.target_soc,
                         self.current_time,
                         self.end_time,
                         self.I_max,
                         self.price
                         ])

        # return np.array([self.current_soc,
        #                  self.target_soc,
        #                  self.start_time,
        #                  self.end_time,
        #                  self.current_time,
        #                  self.current_power_limit,
        #                  self.I_max
        #                  ])

    def step(self, action):
        current = self.current_list[action]
        if current > self.I_max:
            current = self.I_max
        self.current_soc += self.charge_interval * current / self.battery_ah / 60
        self.current_time += 1
        self.current_power_limit = self.get_power_limit()
        self.voltage = self.get_voltage()
        self.I_max = self.get_I_limit()
        self.price = self.get_per_kwh_price()

        observation = np.array([self.current_soc,
                                self.target_soc,
                                self.current_time,
                                self.end_time,
                                self.I_max,
                                self.price
                                ])

        # observation = np.array([self.current_soc,
        #                          self.target_soc,
        #                          self.start_time,
        #                          self.end_time,
        #                          self.current_time,
        #                          self.current_power_limit,
        #                          self.I_max
        #                          ])
        reward = self.get_reward(action)
        terminated = False
        if ((self.current_soc >= self.target_soc) or (self.end_time == self.current_time)):

            terminated = True
            if self.current_soc < self.target_soc:
                # reward += -999
                reward += -abs(self.target_soc - self.current_soc) * self.emission_max_value * self.battery_ah * (
                            1 + self.max_current * self.resistance / self.voltage)

        info = {}

        return observation, reward, terminated, False, info



