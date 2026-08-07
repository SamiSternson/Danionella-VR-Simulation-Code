import numpy as np
import matplotlib.pyplot as plt

powers = [0.5, 1, 1.5, 2]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, p in zip(axes.flatten(), powers):
    probs = [(1 / i**p) for i in range(1, 11)]
    new_probs = [prob / sum(probs) for prob in probs]
    y = np.zeros(10, dtype=int)

    for _ in range(10000):
        y[np.random.choice([i for i in range(10)], p=new_probs)] += 1

    ax.bar([i for i in range(10)], y)
    ax.set_title(f"Power {p}")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")

plt.tight_layout()
plt.show()