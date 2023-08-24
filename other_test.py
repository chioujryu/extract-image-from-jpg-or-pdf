import numpy as np

arr = [[[2339,2,3,5,5], [1653,2,3,5,5], [3,4,5,5,5]]]
arr =  np.array(arr)

print(len(arr.shape))

four_dim_array = np.expand_dims(arr, axis=0)

print(len(four_dim_array.shape))
