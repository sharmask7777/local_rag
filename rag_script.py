import os
os.environ['OPENAI_API_KEY'] = 'dummy'  # Workaround for HuggingFace API key requirement

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp

# Load documents from a directory
loader = DirectoryLoader('./documents', glob="**/*.*")
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Initialize embeddings (using a smaller, faster model for local use)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a vector store
vectorstore = Chroma.from_documents(texts, embeddings)

# Initialize the LLM (adjust path to your local model)
llm = LlamaCpp(model_path="./models/llama-2-7b-chat.Q4_K_M.gguf", n_ctx=2048, n_gpu_layers=0)

# Create a retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Example query
query = "Sample query"
response = qa_chain.run(query)
print(response)
