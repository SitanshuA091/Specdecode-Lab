# **SpedDecode-Lab**

An experimental implementation and exploration of **speculative decoding** for faster autoregressive LLM inference.

## Overview

Speculative decoding uses two language models:

- **Draft model:** A smaller, faster model that proposes multiple tokens.
- **Target model:** A larger model that verifies the proposed tokens in parallel.

The goal is to reduce the number of expensive target-model generation steps while preserving its output distribution.

## What We'll Implement

1. Baseline autoregressive greedy decoding.
2. Draft-token generation using a smaller model.
3. Parallel verification using the target model.
4. Token acceptance and rejection.
5. Probabilistic speculative sampling.

## Experiments

Compare standard decoding against speculative decoding using:

- Tokens generated per second.
- Generation latency.
- Draft-token acceptance rate.
- Performance across different draft lengths and model combinations.

## Scope

This is a learning and research-oriented project focused on understanding and reproducing speculative decoding, not a general-purpose inference framework.

## References

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)