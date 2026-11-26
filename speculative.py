
import argparse
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


TARGET_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32


def load_models():
    print(f"Using device: {DEVICE}")
    print(f"Using dtype: {DTYPE}")

    print("\nLoading target tokenizer...")
    target_tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)

    print("Loading draft tokenizer...")
    draft_tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_ID)

    # Token IDs must mean the same thing for both models.
    if target_tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError(
            "Target and draft tokenizers have different vocabularies. "
            "Cannot compare their token IDs safely."
        )

    special_tokens = [
        "eos_token_id",
        "bos_token_id",
        "pad_token_id",
        "unk_token_id",
    ]

    for token_name in special_tokens:
        if getattr(target_tokenizer, token_name) != getattr(
            draft_tokenizer, token_name
        ):
            raise ValueError(
                f"Tokenizer mismatch: {token_name} differs between models."
            )

    print("\nLoading target model...")
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID,
        dtype=DTYPE,
    ).to(DEVICE)

    print("Loading draft model...")
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_ID,
        dtype=DTYPE,
    ).to(DEVICE)

    target_model.eval()
    draft_model.eval()

    return target_tokenizer, target_model, draft_model

def prepare_prompt(tokenizer, prompt):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Explicitly extract tensors from BatchEncoding.
    input_ids = inputs["input_ids"].to(DEVICE)

    attention_mask = inputs.get("attention_mask")

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(DEVICE)

    return input_ids, attention_mask


@torch.inference_mode()
def generate_draft(
    draft_model,
    prefix_ids,
    draft_length,
    eos_token_id,
):
    attention_mask = torch.ones_like(prefix_ids)

    outputs = draft_model(
        input_ids=prefix_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )

    past_key_values = outputs.past_key_values

    draft_tokens = []

    for step in range(draft_length):

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        draft_tokens.append(next_token)

        # No need to generate further if EOS was produced.
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        # Feed the newly generated token to the draft model.
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (1, 1),
                    dtype=attention_mask.dtype,
                    device=DEVICE,
                ),
            ],
            dim=-1,
        )

        outputs = draft_model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values

    return torch.cat(draft_tokens, dim=-1)

@torch.inference_mode()
def verify_draft(
    target_model,
    prefix_ids,
    draft_token_ids,
):


    draft_length = draft_token_ids.shape[-1]
    prefix_length = prefix_ids.shape[-1]

    combined_ids = torch.cat(
        [prefix_ids, draft_token_ids],
        dim=-1,
    )

    attention_mask = torch.ones_like(combined_ids)

    outputs = target_model(
        input_ids=combined_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    # K + 1 predictions:
    # - First prediction checks draft token 1.
    # - Last prediction is the bonus token if all drafts match.
    target_predictions = outputs.logits.argmax(dim=-1)

    predictions = target_predictions[
        :,
        prefix_length - 1 : prefix_length + draft_length,
    ]

    accepted_tokens = []
    correction_token = None
    bonus_token = None
    all_accepted = True

    for i in range(draft_length):

        proposed = draft_token_ids[0, i].item()
        predicted = predictions[0, i].item()

        if proposed == predicted:
            accepted_tokens.append(proposed)
        else:
            all_accepted = False
            correction_token = predicted
            break

    if all_accepted:
        bonus_token = predictions[0, -1].item()

    return (
        accepted_tokens,
        correction_token,
        bonus_token,
        all_accepted,
    )


@torch.inference_mode()
def speculative_decode(
    tokenizer,
    target_model,
    draft_model,
    prompt,
    max_new_tokens=64,
    draft_length=5,
):
    prompt_ids, _ = prepare_prompt(tokenizer, prompt)

    generated_ids = torch.empty(
        (1, 0),
        dtype=torch.long,
        device=DEVICE,
    )

    eos_token_id = tokenizer.eos_token_id

    total_drafted = 0
    total_accepted = 0
    iteration = 0

    start_time = time.perf_counter()

    while generated_ids.shape[-1] < max_new_tokens:

        iteration += 1

        remaining_tokens = max_new_tokens - generated_ids.shape[-1]

        current_prefix = torch.cat(
            [prompt_ids, generated_ids],
            dim=-1,
        )

        current_draft_length = min(draft_length, remaining_tokens)

        # 1. Draft proposes K tokens.
        draft_token_ids = generate_draft(
            draft_model=draft_model,
            prefix_ids=current_prefix,
            draft_length=current_draft_length,
            eos_token_id=eos_token_id,
        )

        total_drafted += draft_token_ids.shape[-1]

        # 2. Target verifies the proposed tokens.
        (
            accepted_tokens,
            correction_token,
            bonus_token,
            all_accepted,
        ) = verify_draft(
            target_model=target_model,
            prefix_ids=current_prefix,
            draft_token_ids=draft_token_ids,
        )

        total_accepted += len(accepted_tokens)

        # 3. Append the accepted draft prefix.
        if accepted_tokens:
            accepted_tensor = torch.tensor(
                [accepted_tokens],
                dtype=torch.long,
                device=DEVICE,
            )

            generated_ids = torch.cat(
                [generated_ids, accepted_tensor],
                dim=-1,
            )

        print(
            f"Iteration {iteration}: "
            f"{len(accepted_tokens)}/{draft_token_ids.shape[-1]} "
            f"draft tokens accepted"
        )

        # 4. Handle the first rejected token.
        if not all_accepted:

            correction_tensor = torch.tensor(
                [[correction_token]],
                dtype=torch.long,
                device=DEVICE,
            )

            generated_ids = torch.cat(
                [generated_ids, correction_tensor],
                dim=-1,
            )

            if eos_token_id is not None and correction_token == eos_token_id:
                break

        else:
            # All draft tokens matched.
            # Add a bonus token only if generation budget remains.
            if generated_ids.shape[-1] < max_new_tokens:

                bonus_tensor = torch.tensor(
                    [[bonus_token]],
                    dtype=torch.long,
                    device=DEVICE,
                )

                generated_ids = torch.cat(
                    [generated_ids, bonus_tensor],
                    dim=-1,
                )

                if eos_token_id is not None and bonus_token == eos_token_id:
                    break

        # Stop if an accepted draft token was EOS.
        if (
            generated_ids.shape[-1] > 0
            and eos_token_id is not None
            and generated_ids[0, -1].item() == eos_token_id
        ):
            break

    elapsed = time.perf_counter() - start_time

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    acceptance_rate = (
        (total_accepted / total_drafted) * 100
        if total_drafted > 0
        else 0.0
    )

    return (
        generated_text,
        generated_ids,
        total_drafted,
        total_accepted,
        acceptance_rate,
        elapsed,
    )

def main():

    parser = argparse.ArgumentParser(
        description="Greedy speculative decoding using Qwen models."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--draft-length",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")

    if args.draft_length < 1:
        parser.error("--draft-length must be at least 1")

    tokenizer, target_model, draft_model = load_models()

    (
        generated_text,
        generated_ids,
        total_drafted,
        total_accepted,
        acceptance_rate,
        elapsed,
    ) = speculative_decode(
        tokenizer=tokenizer,
        target_model=target_model,
        draft_model=draft_model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        draft_length=args.draft_length,
    )

    print("\n" + "=" * 55)
    print("SPECULATIVE DECODING RESULT")
    print("=" * 55)

    print("\nGenerated response:\n")
    print(generated_text)

    print("\nStatistics:")
    print(f"Generated tokens: {generated_ids.shape[-1]}")
    print(f"Draft tokens proposed: {total_drafted}")
    print(f"Draft tokens accepted: {total_accepted}")
    print(f"Acceptance rate: {acceptance_rate:.2f}%")
    print(f"Generation time: {elapsed:.2f} seconds")

    if elapsed > 0:
        print(
            f"Generation throughput: "
            f"{generated_ids.shape[-1] / elapsed:.2f} tokens/sec"
        )


if __name__ == "__main__":
    main()
