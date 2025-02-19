from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from utils.llm_usage import local_llm


class RAG:
    """
    RAG class for reading PDF files and creating a RetrievalQA chain.

    Methods:
        __init__():
            Initializes the RAG class.

        read_pdf_file(pdf_file):



        retrieve_qa(pdf_read, model_path, chunk_size=100, chunk_overlap=5, model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "mps"}):
    """

    def __init__(self, pdf_file):
        self.pdf_reader = PyMuPDFLoader(pdf_file).load()

    def retrieve_qa(
        self,
        model_path,
        chunk_size=100,
        chunk_overlap=5,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "mps"},
    ):
        """
        Creates a RetrievalQA chain using the provided model and vector database.

        Args:
            model_path: The path to the DeepSeek-R1 model file.
            chunk_size (int, optional): The maximum size of each text chunk. Defaults to 100.
            chunk_overlap (int, optional): The number of characters that overlap between chunks. Defaults to 5.
            model_name (str, optional): The name of the model to use for embedding. Defaults to "sentence-transformers/all-MiniLM-L6-v2".
            model_kwargs (dict, optional): Additional keyword arguments to pass to the model. Defaults to {"device": "mps"}.

        Returns:
            chain: A RetrievalQA chain created using the provided model and vector database.
        """
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
        query = "What engineer is him, and what did he do"
        return qa.invoke(query)


if __name__ == "__main__":
    rag = RAG(pdf_file="Chris_Resume.pdf")
    documents = rag.retrieve_qa(
        model_path="/Users/liaopoyu/Downloads/llm_model/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
        chunk_size=100,
        chunk_overlap=5,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "mps"},
    )
    print(documents)
