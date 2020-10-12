---
title: "Background"
date: 2020-10-01T11:00:57-07:00
draft: false
menu:
  main:
    parent: "discussion"
    name: "Background"
---

# Background for **DiffDART**

This document will attempt to get the intelligent novice up to speed on all the relevant background in order to understand DiffDART.

We'll cover (at a high level):

- What is optimization?
- How does optimization work, and why are gradients useful?
- High level background on neural networks
- What is a physics engine?
- How do physics engines work?
- How do contacts and friction work in a physics engine?

{{< warning >}}
**Warning**: I'm going to oversimplify extremely rich and complex fields. If that offends you, this article isn't for you.
{{< /warning >}}

## What is optimization?

Optimization is about answering questions of the form: "What is the best \_\_\_\_?"

For example:

- "What is the best design for my mechanical component?"
- "What is the best amount to charge for life insurance?"
- "What is the best sequence of controls for my robot?"

Answering questions like this require defining two things:

1. What is the space of all possible options?
2. What do we mean by "best"?

For the space of all possible options, we're going to map an "option" into a `$n$` dimensional vector `$x \in \mathcal{R}^n$`.

**TODO: Illustration of packing a problem into R^n**

For example:

- In mechanical design, our space might be 3 dimensional, where a components length, width, and height are each parameters.
- In life insurance, our space could be a simple 1 dimensional space: what's the annual premium for our life insurance?
- In robotics, our space might be hundreds of dimensions, where each dimension corresponds to the torque applied at a joint for a single millisecond.

Then by "best", we'll just mean minimizing some function `$f(x) \in \mathcal{R}^n \rightarrow \mathcal{R}$`. It's important that `$f(x)$` can only output _a single number_, not a vector. If you have multiple simultaneous objectives, you need to figure out how you want to _weight_ them and then take a weighted sum.

For example:

- In mechanical design, `$f(x)$` might return a weighted sum of the component's mass, and it's ability to withstand loads
- In life insurance, `$f(x)$` might be our expected profit
- In robotics, `$f(x)$` might be some extremely complex function encoding physics and taking a weighted sum over properties of the trajectory, like minimizing energy cost

Once everything is defined, we can write our problem as "Find `$x$` that minimizes `$ f(x)$`". We'll talk about how to do this in the next section.

Once we have our `$x$`, we have our answer to our question "What is the best \_\_\_\_?"

## How does optimization work?

Once we've specified our problem as "Find `$x$` that minimizes `$f(x)$`", all optimization problems begin to yield to the same techniques.

Now we descend from the realm of pure theory to a mixture of theory and practice. We have to use physical computers to solve these optimization problems. This presents a problem, because we have an infinite number of possible `$x$`'s. Yet, each guess for `$x$`, we have to compute the value of `$f(x)$`, and that takes a finite non-zero amount of time.

So we want to find the `$x$` that minimizes `$f(x)$`, and we want to do it in _as few guesses as possible_.

In order to make the problem possible, we're going to make a key assumption about `$f(x)$`: it's _smooth_.

For a small change in `$x$`, we're going to assume we get a small change in `$f(x)$`. That means if a given `$x$` is "pretty good", then a near neighbor `$x + \epsilon$` will also be "pretty good".

**TODO: Illustration of a bowl, with brute force search dots**

So our first strategy for optimization we can think of as "brute force search." Basically, we start with an arbitrary `$x_0$`, then we run the following iteration:

1. Pick a `$j$` random near neighbors around `$x_i$` (there are lots of methods for doing this), call those `$z_0 \ldots z_j$`.
2. Evaluate `$f(z_0), \ldots, f(z_j)$`
3. Terminate if none of the near neighbors improve on `$f(x_i)$`.
4. Set `$x_{i+1} = \text{argmin}_k f(z_k)$`.

This will (inefficiently) descend the current bowl.

The real slowness of "brute force search" comes in step 2, evaluating `$f(z_k)$` for lots of random guesses. If we don't have enough random guesses, our movement won't be in a good direction. If we have too many, we waste time. Either way, it's slow. It'd be great if we didn't have to do this. That brings us to:

### Why are gradients useful?

If we know the _gradient_ of `$f(x)$` (written as `$\nabla f(x)$`), then we can evaluate that function and always know which way to move in our search.

Now our optimization strategy simplifies. Start with an arbitrary `$x_0$`, and we can just move in the direction of `$\nabla f(x)$` until we get to a point where `$\nabla f(x) = 0$`, and then we know we're at the bottom of a bowl.

1. Evaluate `$\nabla f(x_i)$`
2. Terminate if `$\nabla f(x_i) = 0$`
3. Set `$x_{i+1} = x_i + \epsilon * \nabla f(x_i)$`

This is a _much faster_ way to run optimization, which is why gradients are good.

## High level background on neural networks

A "neural network" is a fancy marketing term for a bunch of matrix multiplies, which you tune to do something useful with an optimizer.

Usually people talk about neural networks using block diagrams.

**TODO: simple two layer block diagram**

Here, we've got a very simple network that takes a vector of input, multiplies it by a matrix A, applies an elementwise non-linearity, and multiplies it by a matrix B.

We only specify the size of A and B in advance, and we let an optimizer pick the content of A and B to minimize prediction error over a dataset. So to put this in the language of the previous section, our state is some vector `$x$` that maps its elements to the elements of A and B, and our function `$f(x)$` is a function that runs predictions for our whole dataset and spits out the total error.

The key takeaway here is that neural networks optimize using gradients, not brute force. The state space is absolutely enormous and `$f(x)$` takes a long time to evaluate, so if we didn't have gradients, it'd be _extremely_ difficult to make progress.

Thankfully, we have tools to automatically get the gradients of neural networks. Using Pytorch, the above network would be as easy as:

{{< code python >}}

```
import torch

A = torch.Tensor([3, 100], requires_grad=True)
B = torch.Tensor([100, 1], requires_grad=True)

x = [1, 2, 3]
label = 0.5

guess = B * torch.sigmoid(A * x)
loss = (guess - label)**2
```

{{< /code >}}

The library (Pytorch) keeps track of the graph we used to do our computation, so it can pass gradients back through it later automatically. All we have to do is say

{{< code python >}}

```
loss.backward()

# Now we can read gradients with respect to loss
print(A.grad)
print(B.grad)
```

{{< /code >}}

## What is a physics engine?

A physics engine is a program that simulates the physical world.

Usually, this means that a physics engine is a (fairly complicated) program that can tell you the predicted state of the world in a few milliseconds, if you give in the current state of the world.

You can think of Physics engines as software implementations of familiar Newtonian equations of motion: `$F = ma$`. Except, unlike normal physics, physics engines treat time as discrete, rather than continuous. As an example, the velocity `$v$` and position `$p$` of a point mass `$M$` accelerating under external forces `$F$` might have an update rule like this:

`$v_{t+1} = v_{t} + \Delta t \frac{F}{m}$`

`$p_{t+1} = p_{t} + \Delta t v_{t}$`

Notice how acceleration is implicit, because we're operating in discrete time. A physics engine aggregates lots of these update rules together to form a whole simulated world.

In terms of neural-net style block diagrams, we can think of the physics engine as a single function, predicting the state of the world in a few milliseconds.

![Image](/assets/images/data-flow-fwd.svg)

If the physics engine is a function, it stands to reason that we ought to be able to get gradients through it during a backwards pass. If we could, then we'd be able to include the physics engine into our neural network architectures at arbitrary points.

So this is our goal:

![Image](/assets/images/data-flow-back.svg)

Most of the equations in a physics engine, like the update rules we wrote above for an unconstrained point mass, have easy derivatives and can be automatically differentiated.

The challenge with getting gradients through real physics engines comes from how contacts work in a physics engine.

## How do contacts work in a physics engine?

Contacts are modeled in a physics engine in three steps:

1. Find the contacts (intersections of objects)
2. **Solve an optimization problem** to find a set of contact forces that maintain physical laws
3. Apply those contact forces and run a normal update

The hard part here, from the perspective of differentiation, is dealing with the optimization problem.

**TODO: illustration of triangle collision**

For each contact `$i$`, let's say that `$d_i$` is the relative distance at the contact point. Let's let `$a_i = \ddot{d_i}$` be the acceleration at the contact point. Let `$f_i$` be the force at contact `$i$`.

Our optimization problem is designed to enforce three things:

- No contact points can interpenetrate (`$a_i \geq 0$`)
- All contact forces can only push things apart, not pull them together (`$f_i \geq 0$`)
- No contact force can be applied when a contact is already separating (`$a_if_i = 0$`)

There's a linear relationship between `$f$` and `$a$`. Let's express it as `$Af = a$`, for some matrix `$A$`.

So the write out our contact forces problem, which we must solve at every timestep:

Find `$f$`, `$a$`, such that `$Af = a$`, `$a \geq 0$`, `$f \geq 0$`, and `$a^Tf = 0$`.

It turns out this structure of problem is well studied, and is called a [Linear Complimentarity Problem (LCP)](https://en.wikipedia.org/wiki/Linear_complementarity_problem). These can be solved very efficiently.

Until now, nobody has been able to differentiate through them, because they're technically impossible to differentiate. However, from a practical perspective, we _can_ get gradients through an LCP.

That's in the [next section](/discussion/diff_lcp).
