import matplotlib.pyplot as plt
import numpy as np
import math


def generate_reward(error, resolution=8):
    return ((math.pow(0.3, error)) * 1000).__round__(resolution)


x = []
y = []

for i in np.linspace(0, 30, 300):
    x.append(i)
    y.append(generate_reward(i, 4))

plt.plot(x, y)
plt.grid()
plt.show()
