# Mixture-of-Depths

> Google DeepMind: Mixture-of-Depths Unofficial Implementation.

<div align="center">
<img width="624" alt="MoD" src="https://github.com/Mixture-AI/Mixture-of-Depths/assets/103916249/4e84ca8b-4ace-422e-a8e9-84070a82b14e">
</div>

## TODO List

- [ ] Enable the **batching forward** operation.
- [ ] Static-graph implementation.

## The key of implementation

> Section 3.1. "This is because static-graph implementations account for the worst-case scenarios decisions; e.g., a computation’s inputs will be padded to its capacity amount even if relatively few tokens actually end up routing to it, and/or tokens will be dropped from the computation if the capacity is exceeded."

This section informs us that we need to implement using static graphs and consider the scenarios of padding and dropping. The **padding and dropping** mentioned here refer to the fact that after routing, if the number of tokens routed is less than the capacity, we will pad to the corresponding capacity size, or we will drop tokens that exceed the capacity. (❌ not implement)

> Section 3.4. 
> - "The router weight for a given token embedding is a scalar produced as a result of a linear projection, $r_{i}^{l} = w_\theta^{T} x_{i}^{l}$."
> - "Notably, we multiply the output of the function $f$ by the router weights. This puts the router weights along the “gradient path”, thus subjecting them to the forces of gradient descent through the course of the language modeling task (We experimented with versions where the router weights are also included along the computational path for those tokens that bypass the block’s computations, but it seems to be sufficient—and implementationally simpler—to only include the router weights along the computational path for those tokens that do not bypass the block’s computations)."

This section tells us about the implementation of routing (a single-layer MLP) and how we use and train the router weights. (✅ implement)

> Section 3.5. "The second method introduces a small auxiliary MLP predictor (akin to a second router) that receives the same inputs as the router (with a stop gradient), but whose output is a prediction whether that token will be among the top-k or not in the sequence."

This section informs us how to solve the non-causal problem of the top-k operation. That is, using an auxiliary MLP to predict whether a token will appear in the top-k. (✅ implement)

> Section 5. "If a token does not participate in self-attention at a certain block, then later tokens will also not be able to attend to it."

This section tells us that for any Transformer block, whether a token participates in computation is determined after the first routing and will not change afterward. (✅ implement)
