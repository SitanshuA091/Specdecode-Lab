**python basline.py --prompt "Explain CNNs in brief"**
```bash
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
model.safetensors: downloading bytes: ████████████████████████████████████████████████████████████████| 2.69GB, 9.52MB/s  
model.safetensors: reconstructing file: 100%|████████████████████████████████████████████████| 3.09GB / 3.09GB, 58.8MB/s  
Loading weights: 100%|██████████████████████████████████████████████████████████████████| 338/338 [00:09<00:00, 35.88it/s]
generation_config.json: 100%|█████████████████████████████████████████████████████████████| 242/242 [00:00<00:00, 307kB/s]
[transformers] `past_key_values`/`cache_params` are deprecated. Use `cache` instead.
C:\Scratch_implementations\venv\Lib\site-packages\torch\nn\modules\module.py:1789: FutureWarning: `past_key_values` is deprecated and will be removed in version 5.40.0 for `Qwen2ForCausalLM.forward`. Use `cache` instead.
  return forward_call(*args, **kwargs)

Generated response:

Convolutional Neural Networks (CNNs) are a type of deep learning model that are particularly effective for image and video recognition tasks. They are designed to automatically and adaptively learn spatial hierarchies of features from raw pixel data, making them well-suited for tasks such as image classification, object detection, and segmentation
```

**python draft.py --prompt "Explain RNNs in brief"**

```bash
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█████████████████████████████████████████████████████████████████| 290/290 [00:01<00:00, 267.46it/s]
[transformers] `past_key_values`/`cache_params` are deprecated. Use `cache` instead.
C:\Scratch_implementations\venv\Lib\site-packages\torch\nn\modules\module.py:1789: FutureWarning: `past_key_values` is deprecated and will be removed in version 5.40.0 for `Qwen2ForCausalLM.forward`. Use `cache` instead.
  return forward_call(*args, **kwargs)

Draft output:

RNNs, or

Draft token IDs:
[[49, 9745, 82, 11, 476]]

Generated 5 draft tokens.
```

**python draft.py --prompt "What is a KV cache?" --draft-length 8**
```bash
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█████████████████████████████████████████████████████████████████| 290/290 [00:01<00:00, 260.62it/s]
[transformers] `past_key_values`/`cache_params` are deprecated. Use `cache` instead.
C:\Scratch_implementations\venv\Lib\site-packages\torch\nn\modules\module.py:1789: FutureWarning: `past_key_values` is deprecated and will be removed in version 5.40.0 for `Qwen2ForCausalLM.forward`. Use `cache` instead.
  return forward_call(*args, **kwargs)

Draft output:

A KVM (Key-Value)

Draft token IDs:
[[32, 730, 11187, 320, 1592, 12, 1130, 8]]

Generated 8 draft tokens.
```