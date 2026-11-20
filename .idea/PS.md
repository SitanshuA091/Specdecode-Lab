I'd approach this as a **small inference-engineering project**, not as a model-training project. The interesting part is understanding how two models cooperate during generation and how verification reduces sequential decoding steps.

Since you're currently learning model loading, RAM mapping, `safetensors`, configs and model artifacts, I'd follow this sequence.

## **1. Finish inference fundamentals first**

Before speculative decoding, understand:

- Model architecture: weights, layers, attention, embeddings, LM head.
- Model loading: `config.json`, `model.safetensors`, tokenizer files.
- Hugging Face `from_pretrained()` and what happens internally.
- Autoregressive generation: prompt → logits → sampling → next token.
- KV cache: why previous key/value computations are reused.
- Prefill vs decode: processing the prompt vs generating tokens.

**Checkpoint:** You should be able to load a small pretrained causal LM and implement a basic greedy generation loop yourself.

## **2. Next: Understand speculative decoding**

Read these in order:

1. [**Fast Inference from Transformers via Speculative Decoding (2023)**](https://arxiv.org/abs/2211.17192?utm_source=chatgpt.com) — The foundational paper. Focus on the algorithm and acceptance/rejection methodology.
2. [**Accelerating Large Language Model Decoding with Speculative Sampling (2023)**](https://arxiv.org/abs/2302.01318?utm_source=chatgpt.com) — Important for understanding the probability-based verification and output distribution guarantees.

Focus on just four concepts:

- Draft model proposes \\(K\\) tokens.
- Target model verifies those tokens in parallel.
- Accepted tokens are retained; rejected tokens are corrected using the target distribution.
- A bonus token can be generated when all draft tokens are accepted.

## **3. Choose the models — don't train them**

Use **pretrained models** initially. Training a draft model is a separate research problem and isn't necessary to understand speculative decoding.

My suggested pair:

[Qwen IA - Suite de modelos de lenguaje de Alibaba Cloud](https://images.openai.com/static-rsc-4/wDD9nK6dE97GsQ5vEzRgKYoxUuxEtIY-N12SfyVm4syJFQV4zhNtaHPPwnYKpzJJMrWcA_zCmUVkSYiz21ZlAOspMgo0BNypfB9mjPG5sgkcBkIJ4e74-3-Nw3b9r9X8Z7lnKRFC3orEXKdA_8TjL4dYbsfomT7PiBz26UH2dHc?purpose=inline)

**Draft model**

Qwen2.5-0.5B-Instruct

0.49B parameters · Proposes tokens

[Hugging Face model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?utm_source=chatgpt.com)

[Qwen IA - Suite de modelos de lenguaje de Alibaba Cloud](https://images.openai.com/static-rsc-4/wDD9nK6dE97GsQ5vEzRgKYoxUuxEtIY-N12SfyVm4syJFQV4zhNtaHPPwnYKpzJJMrWcA_zCmUVkSYiz21ZlAOspMgo0BNypfB9mjPG5sgkcBkIJ4e74-3-Nw3b9r9X8Z7lnKRFC3orEXKdA_8TjL4dYbsfomT7PiBz26UH2dHc?purpose=inline)

**Target model**

Qwen2.5-1.5B-Instruct

1.5B parameters · Verifies tokens

[Hugging Face model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct?utm_source=chatgpt.com)

Why this pair?

- Same model family and compatible tokenizer.
- Small enough to experiment with, subject to available memory.
- Both are pretrained, so you can focus on inference.

Verify that their tokenizer vocabularies and token IDs match before implementation.

## **4. What should the project look like?**

I'd build it incrementally.

| Component        | Responsibility                       |
| ---------------- | ------------------------------------ |
| `baseline.py`    | Standard autoregressive generation   |
| `draft.py`       | Generate K candidate tokens          |
| `verifier.py`    | Target model verifies candidates     |
| `speculative.py` | Acceptance, rejection and correction |
| `benchmark.py`   | Measure latency and throughput       |

**Implementation order:**

1. Implement standard greedy decoding using the target model.
2. Implement draft generation (try K = 3, 5, 8).
3. Pass the prompt + draft tokens through the target in one forward pass.
4. Implement greedy verification first:
   - Accept matching draft tokens.
   - At the first mismatch, emit the target's greedy token.
   - If all are accepted, generate one bonus token.
5. Extend to probabilistic rejection sampling using the paper's acceptance and residual-distribution equations.

The greedy implementation is a good starting point, but probabilistic rejection sampling is what gives you the stronger distribution-preservation guarantee.

## **5. What should you measure?**

Your final experiment should compare:

| Metric             | What it tells you                    |
| ------------------ | ------------------------------------ |
| Acceptance rate    | Percentage of draft tokens accepted  |
| Tokens/second      | Actual generation throughput         |
| Speedup            | Speculative vs baseline throughput   |
| Output equivalence | Whether greedy outputs match exactly |

Experiment with different K values and prompt lengths.

One important point: **speculative decoding isn't guaranteed to be faster.** On small models, the overhead of drafting and verification can outweigh the savings. That's an interesting result worth documenting.

## **6. Further exploration (after the project)**

Don't start here, but these are interesting directions:

- [Draft & Verify](https://arxiv.org/abs/2309.08168?utm_source=chatgpt.com) — Self-speculative decoding.
- [Medusa](https://arxiv.org/abs/2401.10774?utm_source=chatgpt.com) — Multiple decoding heads instead of a separate draft model.
- [SpecTr](https://arxiv.org/abs/2310.15141?utm_source=chatgpt.com) — Optimal transport for speculative decoding.

**My recommendation:** Don't rush into coding yet. Finish inference fundamentals, particularly KV caching, prefill/decode, logits and token sampling.

Once those click, speculative decoding becomes a much more understandable engineering problem rather than an implementation copied from a repository.

I'd keep the project PyTorch + Hugging Face Transformers, without vLLM or other inference engines initially. You want to implement the actual draft–verify loop yourself.

I can guide you through the implementation one component at a time once you finish inference basics.&#x20;
