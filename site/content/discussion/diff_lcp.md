---
title: "Differentiating an LCP"
date: 2020-10-01T11:00:57-07:00
draft: false
menu:
  main:
    parent: "discussion"
    name: "Differentiating an LCP"
---

# Differentiating an **LCP**

We assume you're familiar with the [background](/discussion/background). If not, you can read about it [here](/discussion/background).

This is novel work in the process of publication, credit Keenon Werling (keenon@cs.stanford.edu) and Karen Liu (karenliu@cs.stanford.edu).

It turns out that it is possible to get gradients through an LCP in the vast majority of practical scenarios, without recasting it as a QP. To see this, let's consider some hypothetical LCP problem parameterized by `$A, b$` with a solution `$f^{*}$` we've already found.

`$\text{LCP}(A, b) = f^{*}$`

We're interested in finding how our solution changes as we vary `$A$` and `$b$`, `$\frac{\partial f^{*}}{\partial b}$` and `$\frac{\partial f^{*}}{\partial A}$`.

First, let's take note of the structure of the forward solution `$f^{*}$` we already found. Some indices of `$f^{*}$` will be greater than zero, and some will be zero. By complimentarity, we know that if some index `$f^{*}_i > 0$`, then `$a_i = (Af^{*} + b)_i = 0$`. In physical terms, the acceleration at contact point $i$ _must be 0_ if there is any non-zero force being exerted at contact point $i$. Let's call indices like `$i$` "Clamping," because the LCP is changing the force `$f_i > 0$` to keep the acceleration `$a_i = 0$`. Define the set `$\mathcal{C}$` to be all indices that are clamping. Symmetrically, if `$f_j = 0$` for some index `$j$`, then the acceleration `$a_j = (Af^{*} + b)_j \geq 0$` is free to vary without the LCP needing to adjust `$f_j$` to compensate. If `$a_j > 0$` (if `$a_j \neq 0$`) then let's call that "Separating." Define the set `$\mathcal{S}$` to be all indices that are separating. Let's call indices `$j$` where `$a_j = 0$` _and_ `$f_j = 0$` "Floating." Define the set `$\mathcal{F}$` to be all indices that are floating.

If no indices are floating (`$\mathcal{F} = \emptyset$`) then we'll show that the LCP is differentiable. When indices are floating (`$\mathcal{F} \neq \emptyset$`) we'll show that the LCP has several valid subgradients, and it's possible to follow any in an optimization.

### The easy case: no floating indices

First, let's consider the case where `$\mathcal{F} = \emptyset$`. Let's shuffle the indices of `$f^{*}$`, `$a$`, `$A$` and `$b$` to group together members of `$\mathcal{C}$` and `$\mathcal{S}$`. Our LCP becomes:

<div>$$
\begin{matrix}
\text{find} & f^{*}_{\mathcal{C}}, f^{*}_{\mathcal{S}}, a_{\mathcal{C}}, a_{\mathcal{S}}\\
\text{s.t.} &
\begin{bmatrix}
a_{\mathcal{C}} \\
a_{\mathcal{S}}
\end{bmatrix}
=
\begin{bmatrix}
A_{\mathcal{CC}} & A_{\mathcal{CS}} \\
A_{\mathcal{SC}} & A_{\mathcal{SS}} \\
\end{bmatrix}
\begin{bmatrix}
f^{*}_{\mathcal{C}} \\
f^{*}_{\mathcal{S}}
\end{bmatrix} - \begin{bmatrix}
b_{\mathcal{C}} \\
b_{\mathcal{S}}
\end{bmatrix}
\\
& f^{*}_{\mathcal{C}} \geq 0 \\
& f^{*}_{\mathcal{S}} \geq 0 \\
& a_{\mathcal{C}} \geq 0 \\
& a_{\mathcal{S}} \geq 0 \\
& {f^{*}_{\mathcal{C}}}^Ta_{\mathcal{C}} = 0 \\
& {f^{*}_{\mathcal{S}}}^Ta_{\mathcal{S}} = 0 \\
\end{matrix}
$$</div>

Note that we already have a valid solution that satisfies all the constraints, from `$f^{*}$`. And we know which constraints were active. So rewriting the LCP to highlight just the active constraints at our forward solution `$f^{*}$`:

<div>$$
\begin{matrix}
\text{find} && f^{*}_{\mathcal{C}}, f^{*}_{\mathcal{S}}, a_{\mathcal{C}}, a_{\mathcal{S}}\\
\text{s.t.} && 
\begin{bmatrix}
a_{\mathcal{C}}\\
a_{\mathcal{S}}
\end{bmatrix}
=
\begin{bmatrix}
A_{\mathcal{CC}} && A_{\mathcal{CS}}\\
A_{\mathcal{SC}} && A_{\mathcal{SS}}\\
\end{bmatrix}
\begin{bmatrix}
f^{*}_{\mathcal{C}}\\
f^{*}_{\mathcal{S}}
\end{bmatrix}
+
\begin{bmatrix}
b_{\mathcal{C}}\\
b_{\mathcal{S}}
\end{bmatrix}
\\
&& f^{*}_{\mathcal{C}} > 0 \\
&& f^{*}_{\mathcal{S}} = 0 \\
&& a_{\mathcal{C}} = 0 \\
&& a_{\mathcal{S}} > 0 \\
\end{matrix}
$$
</div>

From here it's possible to see how our valid solution `$f^{*}$` changes under infinitesimal perturbations `$\epsilon$` to `$A$` and `$b$`.

To do this, let's define a set of 4 sufficient conditions on `$f^{*}_{\mathcal{C}}$` that, if satisfied, will give a valid solution to the whole perturbed LCP. These are derived from the active constraints above, setting `$f^{*}_{\mathcal{S}} = 0$` and `$a_{\mathcal{C}} = 0$`. These conditions will always be possible to satisfy under small enough perturbations $\epsilon\$ in the neighborhood of the original solution, because the original solution was valid. The clamping indices provide the following constraints:

`$ 0 = A_{\mathcal{CC}} f^{*}_{\mathcal{C}} + b_{\mathcal{C}} $`

`$ f^{*}_{\mathcal{C}} > 0 $`

And the separating indices:

`$ a_{\mathcal{S}} = A_{\mathcal{SC}} f^{*}_{\mathcal{C}} + b_{\mathcal{S}} $`

`$ a_{\mathcal{S}} > 0 $`

Let's first consider tiny perturbations to `$b_{\mathcal{S}}$` and `$A_{\mathcal{SC}}$`. If the perturbations are small enough, then `$a_{\mathcal{S}} > 0$` will still be satisfied with our original `$f_{\mathcal{C}}^{*}$`, because we `$a_{\mathcal{S}} > 0$` holds _strictly_, and so there is some non-zero room to decrease any element `${a_{\mathcal{S}}}_i$` without violating `$a_{\mathcal{S}} > 0$`. That means that:

`$\frac{\partial f^{*}}{\partial b_{\mathcal{S}}} = 0$`
`$\frac{\partial f^{*}}{\partial A_{\mathcal{SC}}} = 0$`

Next let's consider tiny perturbation $\epsilon$ to `$b_{\mathcal{C}}$`. We're interested in the resulting change `$\Delta f^{*}_{\mathcal{C}}$`. We can write our active constraint as:

`$ 0 = A_{\mathcal{CC}} (f^{*}_{\mathcal{C}} + \Delta f^{*}_{\mathcal{C}}) + b_{\mathcal{C}} + \epsilon $`

Rearranging:

`$ A_{\mathcal{CC}} \Delta f^{*}_{\mathcal{C}} = - (A_{\mathcal{CC}} f^{*}_{\mathcal{C}} + b_{\mathcal{C}}) - \epsilon $`

We have that `$A_{\mathcal{CC}} f^{*}_{\mathcal{C}} + b_{\mathcal{C}} = 0$` from our original solution. We also have `$A_{\mathcal{CC}}$` is PSD, because `$A$` was, and is therefore invertible.

`$ \Delta f^{*}_{\mathcal{C}} = - A_{\mathcal{CC}}^{-1}\epsilon $`

This produces a valid solution as long as `$f^{*}_{\mathcal{C}} - A_{\mathcal{CC}}^{-1}\epsilon > 0$`. Since we had that `$f^{*}_{\mathcal{C}} > 0$` with strict equality, it will always be possible to choose an epsilon small enough to make this true. Then we have:

`$\frac{\partial f_{\mathcal{C}}^{*}}{\partial b_{\mathcal{C}}} = -A_{\mathcal{CC}}^{-1}$`

and

`$\frac{\partial f_{\mathcal{S}}^{*}}{\partial b_{\mathcal{C}}} = 0 $`

Last up is handling changes to `$A_{\mathcal{CC}}$`. In practice, changes to `$A_{\mathcal{CC}}$` only happen because we're differentiating with respect to something like body position or link mass, which also changes `$b_{\mathcal{C}}$`. So for our differentiation, we're going to introduce a new scalar variable, `$x$`, which could represent any arbitrary scalar quantity that effects both `$A$` and `$b$`. We're also going to abuse notation, by making `$A_{\mathcal{CC}}$` and `$b_{\mathcal{C}}$` continuous functions which take $x$ as their only argument and return a matrix and vector, respectively.

`$ 0 = A_{\mathcal{CC}}(x) f^{*}_{\mathcal{C}} + b_{\mathcal{C}}(x) $`

We know `$A_{\mathcal{CC}}(x)$` must always be PSD and therefore invertible, so we can get `$f^{*}_{\mathcal{C}}$` in terms of `$x$`:

`$ f^{*}_{\mathcal{C}} = - A_{\mathcal{CC}}(x)^{-1} b_{\mathcal{C}}(x) $`

This will always yield a valid LCP, as long as `$f^{*}_{\mathcal{C}} \geq 0$`. Because `$A_{\mathcal{CC}}(x)$` and `$b_{\mathcal{C}}(x)$` are continuous, and the original solution is valid, then any sufficiently small perturbation to `$x$` will not reduce `$f^{*}_{\mathcal{C}}$` below 0.

Now we can use straightforward matrix calculus to get our Jacobian wrt `$x$`:

`$ \frac{\partial f^{*}_{\mathcal{C}}}{\partial x} = A_{\mathcal{CC}}(x)^{-1}\frac{\partial A_{\mathcal{CC}}(x)}{\partial x} A_{\mathcal{CC}}(x)^{-1}b_{\mathcal{C}}(x) + A_{\mathcal{CC}}(x)^{-1}\frac{\partial b_{\mathcal{C}}(x)}{\partial x} $`

For any specific choice of `$x$`, you'll need to provide `$\frac{\partial A_{\mathcal{CC}}(x)}{\partial x}$` and `$\frac{\partial b_{\mathcal{C}}(x)}{\partial x}$`.

With that, we've got our Jacobians for the LCP when `$\mathcal{F} = \emptyset$`!

### The hard case: floating indices

Now let's consider when `$\mathcal{F} \neq \emptyset$`. Suddenly, the trick from above of replacing the LCP constraints with linear constraints will no longer work, because now we're directionally dependant under perturbations.

Let's say `$i \in \mathcal{F}$`. Consider perturbing `$b_i$` by `$\epsilon$`. If `$\epsilon > 0$`, then `$i$` is immediately bumped into the "separating" set `$\mathcal{S}$`, and `$f^{*}$` doesn't change. If `$\epsilon < 0$`, then `$i$` is immediately bumped into the "clamping" set `$\mathcal{C}$` and `$f^{*}$` must change to compensate.

You can think of elements in `$\mathcal{F}$` as being just like ReLU units that are at exactly 0. The derivative is technically undefined, but it's perfectly fine in a practical optimization to tie break to either of the two subgradients available to you.

Thankfully, encountering elements in `$\mathcal{F}$` is about as rare as encountering ReLU's at exactly 0 in the wild. In simulations, objects almost never touch exactly with no repulsive force. Even setting aside that for numerical reasons exact 0s are rare, to achieve a situation where you'd expect both `$f_i = 0$` and `$a_i = 0$`, you'd need to have no gravity pushing the objects together, and they're need to be perfectly at rest, while being exactly next to each other. This describes books on a bookshelf, if there is absolutely no compressive force pushing the books together. In these cases, you can tie break according to your desired task.
