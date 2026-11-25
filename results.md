

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