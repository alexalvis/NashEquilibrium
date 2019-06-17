import pickle
import numpy as np
import nashpy as nash
from itertools import product
import matplotlib.pyplot as plt

i= 1
v1= []
state = ((0,10),(10,5))
for i in range(2717):
    if i == 0:
        continue
    filename1 = "reward"+ str(i) +".pkl"
    with open(filename1, "rb") as f1:
        reward1 = pickle.load(f1)
    v1.append(reward1[state])
    print(i)

filename2 = "reward791.pkl"
with open(filename2, "rb") as f2:
    reward2 = pickle.load(f2)

# traj = [((6,0),(4,5)),((6,1),(4,5)),((6,2),(4,5)),((6,3),(4,5)),((6,4),(4,5)),((6,5),(4,5)),((6,6),(4,5)),((6,7),(4,5)),((6,8),(4,5)),((6,9),(4,5))]

# v1 = []
# v2 = []
# for state in traj:
#     v1.append(reward1[state])
#     v2.append(reward2[state])
# print(v1)

fig, ax1 = plt.subplots()
plt.plot(range(len(v1)), v1, label='g1', linewidth=2.0, linestyle='--')
# plt.plot(range(len(v2)), v2, label='g2', linewidth=2.0, linestyle='-.')
plt.show()