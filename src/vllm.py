from vllm import LLM


def load_vllm_model_and_tokenizer(model_path):
    llm = LLM(model=model_path)
    return llm, llm.get_tokenizer()
