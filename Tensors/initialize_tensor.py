import torch
import numpy as np

### initializing a Tensor ###

# Directly from data
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

print(x_data)

# From Numpy array:
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# From another tensor:
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

# With random or constant values:
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


### Attributes of Tensors ###

tensor = torch.rand(3,4)

print(f"Shape of tensor : {tensor.shape}")
print(f"Datatype of tensor : {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

### Operations on Tensors ###
# Source : https://pytorch.org/docs/stable/torch.html
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# Standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# Joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


# Arithmetic operations
# Matrix multiplication
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# element-wise product
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)


# single element tensors 
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations Operations that store the result into the operand are called in-place. They are denoted by a _ suffix. For example: x.copy_(y), x.t_(), will change x.

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. Hence, their use is discouraged.

# Bridge with numpy
# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.
# Tensor to numpy array
t = torch.ones(5)
print(f"t: {t}") # t: tensor([1., 1., 1., 1., 1.])

n = t.numpy()
print(f"n: {n}")

# A change in the tensor reflects in the numpy array
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# numpy array to tensor
n = np.ones(5)
t = torch.from_numpy(n)

# Changes in the Numpy array reflects in the tensor
n.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
