# deepseek_langchain
## Overview

The `deepseek_langchain` project aims to enable running the DeepSeek-R1 large language model on a local machine. This project is specifically designed to run on an M3 Max MacBook Pro.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/deepseek_langchain.git
    cd deepseek_langchain
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Download the DeepSeek-R1 model from Hugging Face:
    ```
    https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF
    ```
    
## Requirements

- Python 3.10.15
- The following packages:
    ```sh
    langchain==0.3.18
    langchain-community==0.3.17
    langchain-core==0.3.35
    langchain-huggingface==0.1.2
    langchain-openai==0.3.6
    langchain-text-splitters==0.3.6
    llama_cpp_python==0.3.1
    llamacpp==0.1.14
    openai==1.63.1
    ```

## Usage

To integrate the DeepSeek-R1 model with LangChain and run it locally, follow these steps:

1. Load the model:
    ```python
    from utils.llm_usage import deepseek_r1

    model_path = "models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
    llm = deepseek_r1(model_path)
    ```

2. Run the model:
    ```python
    response = llm("Your input text here")
    print(response)
    ```

3. Execute the script:
    ```sh
    python main.py --model-path models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
    ```

## Goals

Our goal is to seamlessly integrate the DeepSeek-R1 model with LangChain to enable running a local large language model (LLM) on a MacBook Pro.