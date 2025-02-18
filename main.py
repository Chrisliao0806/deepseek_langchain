import argparse
from utils.logger import setup_logging
from utils.llm_usage import deepseek_r1

def parse_arguments():
    parser = argparse.ArgumentParser(description="deepseek r1 model for using langchain")
    parser.add_argument(
        "--model-path",
        default="models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        type=str,
        help="The path to the DeepSeek-R1 model file.",
    )
    parser.add_argument(
        "--question",
        default='｜User｜>What is 1+1?<｜Assistant｜>',
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
    
    llm = deepseek_r1(args.model_path)
    print(llm.invoke(args.question))