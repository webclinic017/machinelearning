'''
import matplotlib.pyplot as plt
tmp = [[1, 2], [3, 1], [2, 0], [7, 6], [4, 9], [8, 4]]
print(tmp[-2:])

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.subplot(132)
plt.subplot(133)
plt.show()
# --------------------------------------------------
import numpy as np

for count, v in enumerate(np.linspace(1, 20, 30)):
    print(count, v)
'''
