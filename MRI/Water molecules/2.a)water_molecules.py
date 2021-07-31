
import math
import random
import numpy as np 
import matplotlib.pyplot as plt
from random import shuffle
import matplotlib.pyplot as plt
d = np.arange(0.0, 160.0, 1.0)
RandomList = [[i] for i in range(1000,1500)]
RandomList2 = [[i] for i in range(43*1000,43*1500)]
plt.suptitle('The random Distribution of magnetic field along z axis on body')
plt.subplot(211)
plt.ylabel('B0 value in mTesla')
plt.xlabel('the distance along z-axis in cm')
shuffle (RandomList)
plt.plot(d,RandomList)
plt.subplot(212)
plt.ylabel('Gyromagnetic ratio of H * B0  mTesla')
plt.xlabel('the distance along z-axis in cm')
RandomList2 = [[i] for i in range(43*1000,43*1500)]
shuffle (RandomList2)
plt.plot(d,RandomList2)

plt.show()