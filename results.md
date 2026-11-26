

### Prompt : RNN
python baseline.py --prompt "Explain RNNs in brief"
```bash
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|██████████████████████████████████████████████████████████████████| 338/338 [00:07<00:00, 42.44it/s]
[transformers] `past_key_values`/`cache_params` are deprecated. Use `cache` instead.
C:\Scratch_implementations\venv\Lib\site-packages\torch\nn\modules\module.py:1789: FutureWarning: `past_key_values` is deprecated and will be removed in version 5.40.0 for `Qwen2ForCausalLM.forward`. Use `cache` instead.
  return forward_call(*args, **kwargs)

Generated response:

Recurrent Neural Networks (RNNs) are a type of artificial neural network designed to recognize patterns in sequences of data, such as time series or speech. They are particularly useful for tasks that involve sequential information, such as language translation, speech recognition, and natural language processing.

Here are the key points about RNN
````
python draft.py --prompt "Explain RNNs in brief" --draft-length 5 
```bash
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [00:02<00:00, 142.70it/s]
[transformers] `past_key_values`/`cache_params` are deprecated. Use `cache` instead.
C:\Scratch_implementations\venv\Lib\site-packages\torch\nn\modules\module.py:1789: FutureWarning: `past_key_values` is deprecated and will be removed in version 5.40.0 for `Qwen2ForCausalLM.forward`. Use `cache` instead.
  return forward_call(*args, **kwargs)

Draft output:

RNNs, or

Draft token IDs:
[[49, 9745, 82, 11, 476]]

Generated 5 draft tokens.
```
python verifier.py --prompt "Explain RNNs in brief" --draft-tokens 49 9745 82 11 476
```bash
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 338/338 [00:07<00:00, 44.96it/s]

Accepted draft tokens:
[]

Accepted text:


Correction / bonus token:
[693]
Re

All draft tokens accepted: False
```
---



### Prompt: KV Cache

python baseline.py --prompt "What is a KV cache?"
```bash
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 338/338 [00:07<00:00, 46.76it/s]
[transformers] `past_key_values`/`cache_params` are deprecated. Use `cache` instead.
C:\Scratch_implementations\venv\Lib\site-packages\torch\nn\modules\module.py:1789: FutureWarning: `past_key_values` is deprecated and will be removed in version 5.40.0 for `Qwen2ForCausalLM.forward`. Use `cache` instead.
  return forward_call(*args, **kwargs)

Generated response:

A KV cache, also known as a key-value cache, is a type of data storage and retrieval system that stores key-value pairs. The key is a unique identifier for each item, and the value is the data associated with that key. KV caches are used to store and retrieve data quickly, and they are often used
```
python draft.py --prompt "What is a KV cache?" --draft-length 8
```bash
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [00:01<00:00, 151.53it/s]
[transformers] `past_key_values`/`cache_params` are deprecated. Use `cache` instead.
C:\Scratch_implementations\venv\Lib\site-packages\torch\nn\modules\module.py:1789: FutureWarning: `past_key_values` is deprecated and will be removed in version 5.40.0 for `Qwen2ForCausalLM.forward`. Use `cache` instead.
  return forward_call(*args, **kwargs)

Draft output:

A KVM (Key-Value)

Draft token IDs:
[[32, 730, 11187, 320, 1592, 12, 1130, 8]]

Generated 8 draft tokens.
```
python verifier.py --prompt "What is a KV cache?" --draft-tokens 32 730 11187 320 1592 12 1130 8
```bash
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 338/338 [00:07<00:00, 44.05it/s]

Accepted draft tokens:
[32]

Accepted text:
A

Correction / bonus token:
[84648]
 KV

All draft tokens accepted: False
```

## COMBINED DRAFTING, VERIFICATION AND GREEDY DECODING
**DRAFT LENGTH = 5**
python speculative.py --prompt "Explain RNNs in brief" --draft-length 5 --max-new-tokens 64

```bash
Using device: cpu
Using dtype: torch.float32

Loading target tokenizer...
Loading draft tokenizer...

Loading target model...
Loading weights: 100%|███████████████████████████████████████████████████████████| 338/338 [00:16<00:00, 21.01it/s]
Loading draft model...
Loading weights: 100%|███████████████████████████████████████████████████████████| 290/290 [00:10<00:00, 28.89it/s]
[transformers] `past_key_values`/`cache_params` are deprecated. Use `cache` instead.
C:\Scratch_implementations\venv\Lib\site-packages\torch\nn\modules\module.py:1789: FutureWarning: `past_key_values` is deprecated and will be removed in version 5.40.0 for `Qwen2ForCausalLM.forward`. Use `cache` instead.
  return forward_call(*args, **kwargs)
Iteration 1: 0/5 draft tokens accepted
Iteration 2: 5/5 draft tokens accepted
Iteration 3: 5/5 draft tokens accepted
Iteration 4: 5/5 draft tokens accepted
Iteration 5: 2/5 draft tokens accepted
Iteration 6: 2/5 draft tokens accepted
Iteration 7: 4/5 draft tokens accepted
Iteration 8: 0/5 draft tokens accepted
Iteration 9: 5/5 draft tokens accepted
Iteration 10: 2/5 draft tokens accepted
Iteration 11: 1/5 draft tokens accepted
Iteration 12: 3/5 draft tokens accepted
Iteration 13: 0/5 draft tokens accepted
Iteration 14: 1/5 draft tokens accepted
Iteration 15: 5/5 draft tokens accepted
Iteration 16: 1/5 draft tokens accepted
Iteration 17: 3/5 draft tokens accepted
Iteration 18: 3/3 draft tokens accepted

=======================================================
SPECULATIVE DECODING RESULT
=======================================================

Generated response:

Recurrent Neural Networks (RNNs) are a type of artificial neural network designed to recognize patterns in sequences of data, such as time series or speech. They are particularly useful for tasks that involve sequential information, such as language translation, speech recognition, and natural language processing.

Here are the key points about RNN

Statistics:
Generated tokens: 64
Draft tokens proposed: 88
Draft tokens accepted: 47
Acceptance rate: 53.41%
Generation time: 115.45 seconds
Generation throughput: 0.55 tokens/sec
```
**DRAFT LENGTH = 8**
python speculative.py --prompt "Explain RNNs in brief" --draft-length 8 --max-new-tokens 64
```bash
Using device: cpu
Using dtype: torch.float32

Loading target tokenizer...
Loading draft tokenizer...

Loading target model...
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 338/338 [00:06<00:00, 49.63it/s]
Loading draft model...
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [00:02<00:00, 108.93it/s]
[transformers] `past_key_values`/`cache_params` are deprecated. Use `cache` instead.
C:\Scratch_implementations\venv\Lib\site-packages\torch\nn\modules\module.py:1789: FutureWarning: `past_key_values` is deprecated and will be removed in version 5.40.0 for `Qwen2ForCausalLM.forward`. Use `cache` instead.
  return forward_call(*args, **kwargs)
Iteration 1: 0/8 draft tokens accepted
Iteration 2: 8/8 draft tokens accepted
Iteration 3: 8/8 draft tokens accepted
Iteration 4: 2/8 draft tokens accepted
Iteration 5: 2/8 draft tokens accepted
Iteration 6: 4/8 draft tokens accepted
Iteration 7: 0/8 draft tokens accepted
Iteration 8: 8/8 draft tokens accepted
Iteration 9: 1/8 draft tokens accepted
Iteration 10: 3/8 draft tokens accepted
Iteration 11: 0/8 draft tokens accepted
Iteration 12: 1/8 draft tokens accepted
Iteration 13: 7/8 draft tokens accepted
Iteration 14: 3/7 draft tokens accepted
Iteration 15: 3/3 draft tokens accepted

=======================================================
SPECULATIVE DECODING RESULT
=======================================================

Generated response:

Recurrent Neural Networks (RNNs) are a type of artificial neural network designed to recognize patterns in sequences of data, such as time series or speech. They are particularly useful for tasks that involve sequential information, such as language translation, speech recognition, and natural language processing.

Here are the key points about RNN

Statistics:
Generated tokens: 64
Draft tokens proposed: 114
Draft tokens accepted: 50
Acceptance rate: 43.86%
Generation time: 77.77 seconds
Generation throughput: 0.82 tokens/sec
```
**DRAFT LENGTH = 3**
python speculative.py --prompt "Explain RNNs in brief" --draft-length 3 --max-new-tokens 64
```bash
Using device: cpu
Using dtype: torch.float32

Loading target tokenizer...
Loading draft tokenizer...

Loading target model...
Loading weights: 100%|██████████████████████████████████████████████████████████████████| 338/338 [00:09<00:00, 37.53it/s]
Loading draft model...
Loading weights: 100%|██████████████████████████████████████████████████████████████████| 290/290 [00:06<00:00, 45.85it/s]
[transformers] `past_key_values`/`cache_params` are deprecated. Use `cache` instead.
C:\Scratch_implementations\venv\Lib\site-packages\torch\nn\modules\module.py:1789: FutureWarning: `past_key_values` is deprecated and will be removed in version 5.40.0 for `Qwen2ForCausalLM.forward`. Use `cache` instead.
  return forward_call(*args, **kwargs)
Iteration 1: 0/3 draft tokens accepted
Iteration 2: 3/3 draft tokens accepted
Iteration 3: 3/3 draft tokens accepted
Iteration 4: 3/3 draft tokens accepted
Iteration 5: 3/3 draft tokens accepted
Iteration 6: 1/3 draft tokens accepted
Iteration 7: 2/3 draft tokens accepted
Iteration 8: 2/3 draft tokens accepted
Iteration 9: 3/3 draft tokens accepted
Iteration 10: 0/3 draft tokens accepted
Iteration 11: 0/3 draft tokens accepted
Iteration 12: 3/3 draft tokens accepted
Iteration 13: 3/3 draft tokens accepted
Iteration 14: 0/3 draft tokens accepted
Iteration 15: 1/3 draft tokens accepted
Iteration 16: 3/3 draft tokens accepted
Iteration 17: 0/3 draft tokens accepted
Iteration 18: 1/3 draft tokens accepted
Iteration 19: 3/3 draft tokens accepted
Iteration 20: 3/3 draft tokens accepted
Iteration 21: 3/3 draft tokens accepted
Iteration 22: 3/3 draft tokens accepted

=======================================================
SPECULATIVE DECODING RESULT
=======================================================

Generated response:

Recurrent Neural Networks (RNNs) are a type of artificial neural network designed to recognize patterns in sequences of data, such as time series or speech. They are particularly useful for tasks that involve sequential information, such as language translation, speech recognition, and natural language processing.

Here are the key points about RNN

Statistics:
Generated tokens: 64
Draft tokens proposed: 66
Draft tokens accepted: 43
Acceptance rate: 65.15%
Generation time: 119.55 seconds
Generation throughput: 0.54 tokens/sec
```