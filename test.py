import numpy as np
import sys
print(sys.version)
print(sys.version_info)
# import tensorflow as tf
a = np.array([[1, 2, 3],[4, 5, 6]])

print(np.prod(a, axis=0))
print(np.prod(a, axis=1))