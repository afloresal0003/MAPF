import time
from math import sqrt
n = 10000000

# For loops
#start = time.time()
#data = []
#for i in range(n):
#    data.append(i**2)
#y = []
#for x in data:
#    data.append(sqrt(x))
#end = time.time()
#print("For loops: {}".format(end-start))

# List comprehension
start = time.time()
data = [x**2 for x in range(n)]
y = [sqrt(x) for x in data]
end = time.time()
print("list comprehension: {}".format(end-start))

# Numpy
import numpy as np
start = time.time()
data = np.arange(n)
y = np.sqrt(data)
end = time.time()
print("numpy: {}".format(end-start))
