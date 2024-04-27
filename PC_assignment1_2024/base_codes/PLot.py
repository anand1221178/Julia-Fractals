import matplotlib.pyplot as plt

# Data
threads = [1, 2, 4, 6, 8, 10, 12, 14, 16]
speedup = [1.08068,2.08742 ,2.19584 ,2.26718,2.79783 ,2.82797,3.34951,3.59085,4.1749 ]

# Plotting
plt.plot(threads, speedup, marker='o', linestyle='-')
plt.title('Speedup vs. Number of Threads')
plt.xlabel('Number of Threads')
plt.ylabel('Speedup')
plt.grid(True)

# Adding labels to the points
for i, (x, y) in enumerate(zip(threads, speedup)):
    plt.text(x, y, f'({x},{y:.4f})', ha='right', va='bottom')

plt.show()
