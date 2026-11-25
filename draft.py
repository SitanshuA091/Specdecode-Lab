import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
).to(device)

model.eval()

@torch.inference_mode()
def generate_draft(prompt, k=5):
    
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        ).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Prefill - process the complete prompt
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )

    past_key_values = outputs.past_key_values

    draft_tokens = []
    draft_distributions = []

    # Generate K candidate tokens.
    for step in range(k):

        logits = outputs.logits[:, -1, :]

        # Convert logits into probability distribution q(x).
        probabilities = torch.softmax(logits, dim=-1)

        # Greedy selection for the initial implementation.
        next_token = torch.argmax(
            probabilities,
            dim=-1,
            keepdim=True,
        )

        draft_tokens.append(next_token)
        draft_distributions.append(probabilities)

        # Stop if EOS (end of sequence/sentence special token) is generated.
        if (
            tokenizer.eos_token_id is not None
            and next_token.item() == tokenizer.eos_token_id
        ):
            break

        # Decode: append the newly generated token.
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

    generated_ids = torch.cat(draft_tokens, dim=-1)

    draft_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    return generated_ids, draft_distributions, draft_text


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate speculative draft tokens"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt",
    )

    parser.add_argument(
        "-k",
        "--draft-length",
        type=int,
        default=5,
        help="Number of candidate draft tokens",
    )

    args = parser.parse_args()

    if args.draft_length < 1:
        parser.error("Draft length must be at least 1.")

    token_ids, distributions, draft_text = generate_draft(
        args.prompt,
        k=args.draft_length,
    )

    print("\nDraft output:\n")
    print(draft_text)

    print("\nDraft token IDs:")
    print(token_ids.tolist())

    print(f"\nGenerated {token_ids.shape[-1]} draft tokens.")
