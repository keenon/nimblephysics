![Stanford Nimble Logo](https://nimblephysics.org/README/README_Splash.svg)

[![Build Status](https://dev.azure.com/keenonwerling/diffdart/_apis/build/status/keenon.nimblephysics?branchName=master)](https://dev.azure.com/keenonwerling/diffdart/_build/latest?definitionId=1&branchName=master)

# Stanford Nimble

`pip3 install nimblephysics`

** BETA SOFTWARE **

[Read our docs](http://www.nimblephysics.org/docs) and [the paper](https://arxiv.org/abs/2103.16021).

Use physics as a non-linearity in your neural network. A single timestep, `nimble.timestep(state, controls)`, is a valid PyTorch function.

![Forward pass illustration](https://nimblephysics.org/README/README_DataFlow_Fwd.svg)

We support an analytical backwards pass, that works even through contact and friction.

![Backpropagation illustration](https://nimblephysics.org/README/README_DataFlow_Back.svg)

It's as easy as:

```python
from nimble import timestep

# Everything is a PyTorch Tensor, and this is differentiable!!
next_state = timestep(world, current_state, control_forces)
```

Nimble started life as a fork of the popular DART physics engine, with analytical gradients and a PyTorch binding. We've worked hard to maintain as much backwards compatability as we can, so many simulations that worked in DART should translate directly to Nimble.

Check out our [website](http://www.nimblephysics.org) for more information.
