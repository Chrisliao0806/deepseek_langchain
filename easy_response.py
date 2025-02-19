import argparse
from utils.logger import setup_logging
from utils.llm_usage import local_llm


def parse_arguments():
    """
    Parses command-line arguments for the DeepSeek-R1 model using langchain.

    Returns:
        argparse.Namespace: A namespace containing the parsed command-line arguments:
            - model_path (str): The path to the DeepSeek-R1 model file. Default is "models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf".
            - question (str): The question to ask the model. Default is "｜User｜>What is 1+1?<｜Assistant｜>".
            - nctx (int): The number of tokens to output for context. Default is 512.
            - max_tokens (int): The maximum number of tokens to generate. Default is 512.
    """
    parser = argparse.ArgumentParser(
        description="deepseek r1 model for using langchain"
    )
    parser.add_argument(
        "--model-path",
        default="models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        type=str,
        help="The path to the DeepSeek-R1 model file.",
    )
    parser.add_argument(
        "--question",
        default="｜User｜>What is 1+1?<｜Assistant｜>",
        type=str,
        help="The question to ask the model.",
    )
    parser.add_argument(
        "--nctx",
        default=512,
        type=int,
        help="The number of tokens to output for context.",
    )
    parser.add_argument(
        "--max-tokens",
        default=512,
        type=int,
        help="The maximum number of tokens to generate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    setup_logging("INFO")

    llm = local_llm(args.model_path)
    print(llm.invoke(args.question))
