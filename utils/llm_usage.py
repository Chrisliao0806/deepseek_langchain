from langchain_community.llms import LlamaCpp


def deepseek_r1(model_path, max_tokens=512):
    """
    Load the DeepSeek-R1 model and return the Llama object.

    Args:
        model_path (str): The path to the DeepSeek-R1 model file.
        context_output (int): The number of tokens to output for context.

    Returns:
        Llama: The Llama object for the DeepSeek-R1 model.
    """
    llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1,
    temperature=0.3,
    n_ctx=2048,
    max_tokens=max_tokens,
    top_p=1,
    f16_kv=True,
    verbose=True,  # Verbose is required to pass to the callback manager
)
    return llm
    