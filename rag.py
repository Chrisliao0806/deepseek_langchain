import argparse
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from utils.llm_usage import local_llm


def parse_arguments():
    """
    Parses command-line arguments for the RAG model using langchain.

    Returns:
        argparse.Namespace: The parsed command-line arguments.

    Arguments:
        --pdf-file (str): The path to the PDF file. Default is "Chris_Resume.pdf".
        --model-path (str): The path to the DeepSeek-R1 model file. Default is "/Users/liaopoyu/Downloads/llm_model/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf".
        --chunk-size (int): The maximum size of each text chunk. Default is 100.
        --chunk-overlap (int): The number of characters that overlap between chunks. Default is 5.
        --model-name (str): The name of the model to use for embedding. Default is "sentence-transformers/all-MiniLM-L6-v2".
        --gpu-usage (str): Additional keyword arguments to pass to the model, such as "cpu", "gpu". Defaults to "mps".
        --question (str): The question to ask the model. Default is "What engineer is him, and what did he do".
    """
    parser = argparse.ArgumentParser(description="RAG model for using langchain")
    parser.add_argument(
        "--pdf-file",
        default="Chris_Resume.pdf",
        type=str,
        help="The path to the PDF file.",
    )
    parser.add_argument(
        "--model-path",
        default="/Users/liaopoyu/Downloads/llm_model/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
        type=str,
        help="The path to the DeepSeek-R1 model file.",
    )
    parser.add_argument(
        "--chunk-size",
        default=300,
        type=int,
        help="The maximum size of each text chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        default=10,
        type=int,
        help="The number of characters that overlap between chunks.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        type=str,
        help="The name of the model to use for embedding.",
    )
    parser.add_argument(
        "--gpu-usage",
        default="mps",
        type=str,
        help='Additional keyword arguments to pass to the model,such as "cpu", "gpu". Defaults to "mps".',
    )
    parser.add_argument(
        "--question",
        default="What engineer is him, and what did he do",
        type=str,
        help="The question to ask the model.",
    )
    return parser.parse_args()


class RAG:
    """
    RAG class for reading PDF files and creating a RetrievalQA chain.

    Methods:
        __init__():
            Initializes the RAG class.
        retrieve_qa(pdf_read, model_path, chunk_size=100, chunk_overlap=5, model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "mps"}):
    """

    def __init__(self, pdf_file):
        self.pdf_reader = PyMuPDFLoader(pdf_file).load()

    def retrieve_qa(
        self,
        model_path,
        query,
        chunk_size=300,
        chunk_overlap=10,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        gpu_usage="mps",
    ):
        """
        Creates a RetrievalQA chain using the provided model and vector database.

        Args:
            model_path: The path to the DeepSeek-R1 model file.
            chunk_size (int, optional): The maximum size of each text chunk. Defaults to 100.
            chunk_overlap (int, optional): The number of characters that overlap between chunks. Defaults to 5.
            model_name (str, optional): The name of the model to use for embedding. Defaults to "sentence-transformers/all-MiniLM-L6-v2".
            gpu_usage (str, optional): Additional keyword arguments to pass to the model,such as "cpu", "gpu". Defaults to "mps".

        Returns:
            chain: A RetrievalQA chain created using the provided model and vector database.
        """
        model_kwargs = {}
        model_kwargs["device"] = gpu_usage
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        all_splits = text_splitter.split_documents(self.pdf_reader)

        # Embed text
        embedding = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs
        )

        # Create Chroma vector database
        vectordb = Chroma.from_documents(
            documents=all_splits, embedding=embedding, persist_directory="db"
        )

        # Create RetrievalQA chain
        llm = local_llm(model_path)
        retriever = vectordb.as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, verbose=True
        )
        return qa.invoke(query)


if __name__ == "__main__":
    args = parse_arguments()
    rag = RAG(pdf_file="Chris_Resume.pdf")
    documents = rag.retrieve_qa(
        model_path=args.model_path,
        query=args.question,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model_name,
        gpu_usage=args.gpu_usage,
    )
    print(documents)
