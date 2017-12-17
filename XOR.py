import numpy as np

# Sigmoid
def sigmoid(i, isDerivative = False):
  return i * (1 - i) if isDerivative else 1 / (1 + np.exp(-i))
  
# Input matrix
input_matrix = np.array([
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],    
    [1,0,1],
    [1,1,0],
    [1,1,1],
    ])

# XOR output:
correct_output = np.array([
    [0],
    [0],    
    [1],
    [0],
    [1],
    [1],
    [0],
    ])

# Random generator seed.
np.random.seed(1)

#synapses
s1 = 2 * np.random.random((3,8)) - 1
s2 = 2 * np.random.random((8,1)) - 1  

#training step
for i in range(100000):  
    
    # layers
    base_layer  = input_matrix
    mid_layer   = sigmoid(np.dot(base_layer, s1))
    final_layer = sigmoid(np.dot(mid_layer, s2))
    
    # Back propagation
    final_error = correct_output - final_layer
    final_delta = final_error * sigmoid(final_layer, True)
    mid_error   = final_delta.dot(s2.T)
    mid_delta   = mid_error * sigmoid(mid_layer, True)
    
    # Prints every 1000 steps
    if i % 1000 == 0: 
      error_rate = np.mean(np.abs(final_error))
      print("Error rate: " + str(error_rate))
        
    # Update synapses
    s1 += base_layer.T.dot(mid_delta)
    s2 += mid_layer.T.dot(final_delta)  

print("Post-training output:")
print(final_layer)