import torch
import torchax as tx
import jax
import jax.numpy as jnp

# the torchax environment is what make pytorch operators work on torchax's Tensor
env = tx.default_env()

# Start with an jax array:

arr = jnp.ones((4, 4))

# call torch function on jax array is error
# torch.matmul(arr, arr)

# convert arr to a Tensor
tensor = tx.interop.torch_view(arr)

print('Is torch Tensor:', isinstance(tensor, torch.Tensor)) # prints True
print('inner data of my tensor', tensor.__dict__)

# Running pytorch operators on top of our custom tensor
print('demo 1: ====')
with env:
  print(torch.matmul(tensor, tensor))
  print(torch.sin(tensor))
  print(torch.exp(tensor))

print('demo 2: ====')
with env:
  tensor = torch.ones( (4,4), device='jax')
  print(torch.matmul(tensor, tensor))
  print(torch.sin(tensor))
  print(torch.exp(tensor))

print('====')