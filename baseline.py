import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
).to(device)

model.eval()


@torch.inference_mode()
def greedy_decode(prompt, max_new_tokens=64):

    messages = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    attention_mask = torch.ones_like(input_ids)

    # PREFILL: Process the entire prompt.
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )

    past_key_values = outputs.past_key_values

    # Predict the first token.
    next_token = outputs.logits[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    generated = []

    # DECODE: Process one new token at a time.
    for step in range(max_new_tokens):

        generated.append(next_token)

        if (
            tokenizer.eos_token_id is not None
            and next_token.item() == tokenizer.eos_token_id
        ):
            break

        if step == max_new_tokens - 1:
            break

        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)],
            dim=-1,
        )

        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values

        next_token = outputs.logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

    generated_ids = torch.cat(generated, dim=-1)

    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline greedy decoding"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for the model",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    result = greedy_decode(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    print("\nGenerated response:\n")
    print(result)
