"""
Design and conduct an experiment to demonstrate the difficulties that sample-average methods have
for nonstationary problems. Use a modified version of the 10-armed testbed in which all q*(a) start
out equal and then take independent random walks. Prepare plots for an action-value method using
sample averages, incrementally computed by alpha = 1 / n, and another action-value method using a
constant step-size parameter, alpha = 0.1. Use eps = 0.1 and, if necessary, runs longer than 1000
steps.
"""

import numpy as np
import matplotlib.patches as mpatches  # In order to add legend to the plot.
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
sns.set(style="darkgrid")

k = 10
alpha = 0.1
eps = 0.1

n_reps = 100
n_steps = 5001

rewards_sa = np.zeros([n_reps, n_steps])  # Sample average.
rewards_cs = np.zeros([n_reps, n_steps])  # Constant step.

q_star_init = np.random.randn()
q_star_bandits = np.zeros([n_steps, k])
q_star_bandits[0] = q_star_init
for i in range(1, n_steps):
    q_star_bandits[i] = q_star_bandits[i - 1] + 0.1 * np.random.randn(k)

for rep in range(n_reps):
    sample_average_est = np.array([[0, 0] for i in range(k)])
    const_step_est = np.random.randn(k)  # Random initialization.

    for step in range(n_steps):
        # Sample average.
        if step < k:
            act = step
        elif np.random.rand() < eps:
            act = np.random.randint(0, 10)
        else:
            act = np.argmax(sample_average_est[:, 0])
        r = np.random.randn() + q_star_bandits[step, act]
        rewards_sa[rep][step] = r

        # Update.
        sample_average_est[act][1] += 1
        sample_average_est[act][0] += (r - sample_average_est[act][0]) / sample_average_est[act][1]

        # Constant step.
        if np.random.rand() < eps:
            act = np.random.randint(0, 10)
        else:
            act = np.argmax(const_step_est)
        r = np.random.randn() + q_star_bandits[step, act]
        rewards_cs[rep][step] = r

        # Update.
        const_step_est[act] += alpha * (r - const_step_est[act])

sns.tsplot(data=rewards_sa, color='b')
sns.tsplot(data=rewards_cs, color='r')
ax = sns.tsplot(data=np.max(q_star_bandits, axis=1), color='g')

p1 = mpatches.Patch(color='b', label='Sample average')
p2 = mpatches.Patch(color='r', label='Constant step')
p3 = mpatches.Patch(color='g', label='q*_max')
plt.legend(handles=[p1, p2, p3])

ax.set(xlabel='Steps', ylabel='Average reward')
sns.plt.show()