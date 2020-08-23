import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from statistics import mean, median
from collections import Counter
env = gym.make("Pendulum-v0")
env.reset()

for _ in range(300):
    env.render()
    action = random.uniform(-2, 2)
    print(action)
    env.step([action])