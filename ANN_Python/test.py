import matplotlib.pyplot as plt
import numpy as np

time = np.arange(0.0, 3500.0, 10.0)
goals = []
for minutes in time:
    goals.append(3.18*10**(-8) * minutes**3 - 0.000081088 * minutes**2 + 0.19792 * minutes - 8.3594)
line1, = plt.plot(time, goals, label="Goals scored")
plt.xlabel("Time passed (minutes)")
plt.ylabel("Field goals scored")
# plt.legend(handles=[line1])
plt.show()
