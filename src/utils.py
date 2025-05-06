import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

def load_model_and_tokenizer(model_path, device="cuda:0"):

    config = AutoConfig.from_pretrained(model_path)
    """
    if device == "cpu":
        if 'loss_type' in config.__dict__:
            del config.loss_type
    """
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

"""
Checkpoint folder format is like : result_gpt2_t0-<...sub-task-name..>
"""
def list_all_models_and_datasets(
    checkpoint_folder="checkpoints/", model="gpt2", info=False
):
    model_paths = []
    dataset_paths = []
    subtask_type = []
    for ckpt in os.listdir(checkpoint_folder):
        if model in ckpt:
            # this is a checkpoint from the model named gpt2
            model_paths.append(os.path.join(checkpoint_folder, ckpt))
            prefix_len = len(f"result_{model}_")
            sub_task_name = ckpt[prefix_len : ckpt.find("-")]
            dataset_paths.append(os.path.join("data", sub_task_name, ckpt[prefix_len:]) + ".json")
            subtask_type.append(sub_task_name)

    if info:
        print("Number of tasks Identified : ", len(model_paths))
        print("Number of unique subtasks identified : ", len(set(subtask_type)))

    return subtask_type, model_paths, dataset_paths
