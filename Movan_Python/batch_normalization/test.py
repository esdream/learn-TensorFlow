
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:

plt.axis([0, 10, 0, 1])
# 交互绘图
plt.ion()

for i in range(10):
    y = np.random.random()
    plt.scatter(i, y)
    plt.pause(0.05)
    
while True:  
    plt.pause(0.05)


# In[ ]:



