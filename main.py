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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    setup_logging("INFO")
    
    llm = deepseek_r1(args.model_path)