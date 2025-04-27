import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams


def load_json(path):
    """Load a JSON file where each line has keys 'prompt' and 'response'. (only loads TRAIN split)"""
    with open(path, "r") as f:
        data = json.load(line)["train"]

    return data


def compute_log_probs_and_lens(llm, tokenizer, dataset, batch_size=16):
    """
    For each (prompt, response) in `dataset`, returns two numpy arrays:
      - logps: scalar log P(y|x) summed over response tokens
      - resp_lens: number of tokens in each response
    """
    all_logps = np.zeros(len(dataset))
    all_lens = np.zeros(len(dataset), dtype=np.int32)

    # Pre-process the dataset to prepare all prompts and responses at once
    print("Preprocessing texts...")
    all_texts = []
    all_prefix_lens = []
    all_resp_lens = []

    for ex in tqdm(dataset):
        p = ex["prompt"] + tokenizer.eos_token
        r = ex["response"]
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        r_ids = tokenizer.encode(r, add_special_tokens=False)
        all_prefix_lens.append(len(p_ids))
        all_resp_lens.append(len(r_ids))
        all_texts.append(p + r)

    # Process in batches with progress tracking
    print("Computing log probabilities...")
    for start in tqdm(range(0, len(dataset), batch_size)):
        end = min(start + batch_size, len(dataset))
        batch_texts = all_texts[start:end]
        batch_prefix_lens = all_prefix_lens[start:end]
        batch_resp_lens = all_resp_lens[start:end]

        # Get per-token logprobs for entire inputs
        sampling_params = SamplingParams(
            prompt_logprobs=0,  # return logprob per input token
            max_tokens=1,  # do not generate new tokens
        )
        outputs = llm.generate(batch_texts, sampling_params=sampling_params)

        # Process outputs efficiently
        for i, (out, p_len, r_len) in enumerate(
            zip(outputs, batch_prefix_lens, batch_resp_lens)
        ):
            idx = start + i
            if out.prompt_logprobs:
                resp_slice = out.prompt_logprobs[p_len : p_len + r_len]
                lp = sum(list(tok.values())[0].logprob for tok in resp_slice)
            else:
                lp = float("-inf")
            all_logps[idx] = lp
            all_lens[idx] = r_len

    return all_logps, all_lens


def main(model_paths, dataset_paths, batch_size=16, alpha=1.0, cache_results=True):
    num_models = len(model_paths)
    num_datasets = len(dataset_paths)

    print(f"Processing {num_models} models on {num_datasets} datasets...")

    # Load all datasets once
    print("Loading datasets...")
    datasets = [load_json(p) for p in dataset_paths]
    N = np.array([len(d) for d in datasets])

    # Initialize arrays
    L = np.zeros((num_models, num_datasets))
    H = np.zeros((num_models, num_datasets))

    # Create cache system to avoid recomputing results
    cache_files = {}
    if cache_results:
        os.makedirs("cache", exist_ok=True)
        for i, model_path in enumerate(model_paths):
            model_name = os.path.basename(model_path)
            for j, dataset_path in enumerate(dataset_paths):
                dataset_name = os.path.basename(dataset_path)
                cache_file = (
                    f"cache/model_{i}_{model_name}_dataset_{j}_{dataset_name}.npz"
                )
                cache_files[(i, j)] = cache_file

    # Process each model
    for i, model_path in enumerate(model_paths):
        print(f"Loading model {i} from {model_path} ...")
        llm = LLM(model=model_path, enable_prefix_caching=True)
        tokenizer = llm.get_tokenizer()

        for j, data in enumerate(datasets):
            # Check if results are already cached
            if (
                cache_results
                and (i, j) in cache_files
                and os.path.exists(cache_files[(i, j)])
            ):
                print(f"  → Loading cached results for model {i} on task {j}")
                cached = np.load(cache_files[(i, j)])
                logps = cached["logps"]
                resp_lens = cached["resp_lens"]
            else:
                print(f"  → Eval model {i} on task {j} ({N[j]} examples)")
                logps, resp_lens = compute_log_probs_and_lens(
                    llm, tokenizer, data, batch_size
                )

                # Cache results
                if cache_results:
                    np.savez(cache_files[(i, j)], logps=logps, resp_lens=resp_lens)

            # Vectorized operations for efficiency
            L[i, j] = np.sum(logps)
            H[i, j] = np.sum(-logps / resp_lens) / N[j]

        # Free memory explicitly
        del llm
        torch.cuda.empty_cache()

    # Compute H_j^min for each task j
    H_min = np.min(H, axis=0)

    # Build similarity matrix S efficiently
    S = np.zeros((num_models, num_datasets))
    for j in range(num_datasets):
        # Vectorized computation where possible
        diffs = (L[:, j] - L[j, j]) / (2 * N[j])
        bases = np.exp(-diffs)
        gates = np.exp(-alpha * (H[:, j] - H_min[j]))
        S[:, j] = bases  # * gates

    # Print LaTeX matrix
    print("\n% Pairwise similarity kernel S_{ij} with cross-entropy gate")
    print(r"\begin{pmatrix}")
    for j in range(num_datasets):
        row = [f"{S[i, j]:.4f}" for i in range(num_models)]
        print("  " + " & ".join(row) + r" \\")
    print(r"\end{pmatrix}")

    # Save final results for further analysis
    np.savez("similarity_results.npz", L=L, H=H, S=S)
    print("Final results saved to similarity_results.npz")


# --- 1) Create two tiny JSONL hold-out files ---
temp_dir = Path(tempfile.mkdtemp())
datasets = {
    "task1": [{"prompt": "Hello, how are you?", "response": "I'm fine, thanks!"}],
    "task2": [{"prompt": "What's 2 + 2?", "response": "The answer is 4."}],
}

dataset_paths = []
for task_name, examples in datasets.items():
    fp = temp_dir / f"{task_name}_holdout.jsonl"
    with fp.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    dataset_paths.append(str(fp))

# --- 2) Use GPT-2 as a stand-in for two “models” ---
model_paths = [
    "openai-community/gpt2",
    "facebook/opt-6.7b",
]  # in real use these would be local dirs

# --- 3) Run the full similarity-kernel computation on this toy data ---
#    batch_size=1 so it’s easy to see
main(model_paths, dataset_paths, batch_size=1)
