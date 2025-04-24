import torch
from torch.utils.data import DataLoader

from src.preprocess.dataset import (
    DataCollatorForInstructionTuning,
    load_dataset,
    preprocess_dataset,
)
from src.utils import list_all_models_and_datasets
from src.vllm import load_vllm_model_and_tokenizer
from vllm import SamplingParams


# test dataset name :  t0-task-sciq_Multiple_Choice.json
def compute_label_probabilities(
    llm,
    tokenizer,
    dataset,
):
    """
    For a given dataset, compute for each sample the probability assigned by the model
    to the ground-truth label.
    """
    label_probs = []
    sampling_params = SamplingParams(prompt_logprobs=0, max_tokens=1)

    for sample in dataset:
        batch_input_ids = sample["input_ids"]  # shape = (batches, seq_length)
        outputs = llm.generate(
            prompt_token_ids=batch_input_ids, sampling_params=sampling_params
        )

        print(outputs)

    return label_probs


def compute_similarity_score(probs_A_on_B, probs_B_on_B, probs_B_on_A, probs_A_on_A):
    """
    Computes the PMI similarity score between tasks using the probabilities for the ground-truth labels.

    For samples from dataset D_Tj:
      term1 = (1/n) sum[ log( p_A(y|x) / p_B(y|x) ) ]

    For samples from dataset D_Ti:
      term2 = (1/m) sum[ log( p_B(y|x) / p_A(y|x) ) ]

    Finally:
      S_AB = 0.5 * (term1 + term2)
    """
    n = len(probs_A_on_B)
    m = len(probs_B_on_A)

    term1 = (
        sum(
            torch.log(torch.tensor(pA) / torch.tensor(pB))
            for pA, pB in zip(probs_A_on_B, probs_B_on_B)
        )
        / n
    )
    term2 = (
        sum(
            torch.log(torch.tensor(pB) / torch.tensor(pA))
            for pA, pB in zip(probs_A_on_A, probs_B_on_A)
        )
        / m
    )

    S_AB = 0.5 * (term1 + term2)
    return S_AB.item()


subtask_types, model_paths, dataset_paths = list_all_models_and_datasets(
    checkpoint_folder="checkpoints/full_ft"
)

task = 10
dataset = 13
llm, tokenizer = load_vllm_model_and_tokenizer(model_paths[task], None)

data_collator = DataCollatorForInstructionTuning(tokenizer)

d = load_dataset(dataset_paths[dataset])

pd = preprocess_dataset(d, tokenizer, None)

train_dataloader = DataLoader(
    pd,
    shuffle=False,
    collate_fn=data_collator,
    batch_size=3,
    pin_memory=True,
    num_workers=8,
)
compute_label_probabilities(llm, tokenizer, train_dataloader, "cuda:0")
