import os
import sys
import argparse
import yaml
from typing import Dict, Any, Optional, List, Tuple

# Disable ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['OPENAI_API_KEY'] = 'dummy'  # Workaround for HuggingFace API key requirement

from langchain_community.document_loaders import DirectoryLoader
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import LlamaCpp
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# Monkey patch to fix LlamaCpp cleanup issues
try:
    import llama_cpp.llama
    
    # Store the original __del__ method
    original_del = llama_cpp.llama.Llama.__del__
    
    # Create a safer __del__ method that doesn't crash
    def safe_del(self):
        try:
            if hasattr(self, 'ctx') and self.ctx is not None:
                self.ctx = None
        except:
            pass
    
    # Replace the __del__ method
    llama_cpp.llama.Llama.__del__ = safe_del
    print("Applied safer cleanup for LlamaCpp")
except Exception as e:
    print(f"Warning: Could not patch LlamaCpp cleanup: {e}")

def safe_filter_metadata(doc: Document) -> Document:
    """
    Safely filter complex metadata from a document.
    Only keep simple types (str, int, float, bool) in metadata.
    """
    if not hasattr(doc, 'metadata'):
        return doc
    
    safe_metadata = {}
    for key, value in doc.metadata.items():
        if isinstance(value, (str, int, float, bool)):
            safe_metadata[key] = value
    
    return Document(page_content=doc.page_content, metadata=safe_metadata)

def load_config() -> Dict[str, Any]:
    """Load configuration with sensible defaults for M2 Pro with 16GB RAM."""
    default_config = {
        "model": {
            "path": "./models/llama-2-7b-chat.Q4_K_M.gguf",  # Default to existing model
            "context_size": 4096,
            "gpu_layers": -1,
            "use_mlock": True,
            "f16_kv": True
        },
        "embedding": {
            "model_name": "BAAI/bge-small-en-v1.5",
            "use_gpu": True
        },
        "vectordb": {
            "type": "chroma",
            "location": "./chroma_db",
            "collection_name": "documents"
        },
        "documents": {
            "directory": "./documents",
            "chunk_size": 500,
            "chunk_overlap": 100
        },
        "retrieval": {
            "k": 4,
            "use_hybrid_search": True
        }
    }
    
    # Create config file if it doesn't exist
    if not os.path.exists("config.yaml"):
        os.makedirs(os.path.dirname("config.yaml"), exist_ok=True)
        with open("config.yaml", "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)
        print("Created default config.yaml file")
    
    # Load from file if it exists
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            print("Loaded configuration from config.yaml")
            return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        print("Using default configuration")
        return default_config

def initialize_qa_system() -> Tuple[ConversationalRetrievalChain, LlamaCpp]:
    """Initialize the QA system with components optimized for M2 Pro."""
    config = load_config()
    
    print("Loading documents...")
    # Create the document directory if it doesn't exist
    os.makedirs(config["documents"]["directory"], exist_ok=True)
    
    # Load documents from a directory with better handling
    try:
        # Use UnstructuredFileLoader for each file to avoid pickling issues
        documents = []
        for file_path in os.listdir(config["documents"]["directory"]):
            if file_path.endswith('.pdf'):
                full_path = os.path.join(config["documents"]["directory"], file_path)
                try:
                    print(f"Loading {file_path}...")
                    loader = UnstructuredLoader(full_path)
                    docs = loader.load()
                    
                    # Ensure each document has proper metadata
                    valid_docs = []
                    for doc in docs:
                        if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                            # Create a new document with empty metadata if needed
                            doc = Document(page_content=doc.page_content, metadata={})
                        valid_docs.append(doc)
                    
                    documents.extend(valid_docs)
                    print(f"Successfully loaded {len(valid_docs)} elements from {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
        
        print(f"Loaded {len(documents)} documents total")
    except Exception as e:
        print(f"Error loading documents: {e}")
        documents = []
    
    if not documents:
        print("No documents found or unable to load documents.")
        print(f"Please ensure documents are available in {config['documents']['directory']}")
        sys.exit(1)
    
    # Split documents into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["documents"]["chunk_size"],
        chunk_overlap=config["documents"]["chunk_overlap"],
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    print("Splitting documents into chunks...")
    try:
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks")
    except Exception as e:
        print(f"Error splitting documents: {e}")
        print("Falling back to unsplit documents")
        texts = documents
    
    # Initialize optimized embeddings for M2 Pro
    print("Initializing embedding model...")
    embedding_config = config["embedding"]
    device = "mps" if embedding_config["use_gpu"] else "cpu"
    
    # Use the new import path for HuggingFaceEmbeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_config["model_name"],
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create Chroma vector database for persistent storage
    print("Creating vector database...")
    vectordb_config = config["vectordb"]
    
    # Create directory for Chroma database if it doesn't exist
    os.makedirs(vectordb_config["location"], exist_ok=True)
    
    # Create or connect to existing collection
    try:
        # First, delete existing collection if it exists
        try:
            print("Checking for existing vector database...")
            client = Chroma(
                persist_directory=vectordb_config["location"],
                embedding_function=embeddings
            )
            client.delete_collection(vectordb_config["collection_name"])
            print("Deleted existing collection")
        except Exception as e:
            print(f"No existing collection found or {e}")
        
        # Filter complex metadata from documents before creating vector store
        print("Filtering complex metadata from documents...")
        filtered_texts = []
        for doc in texts:
            try:
                filtered_doc = safe_filter_metadata(doc)
                filtered_texts.append(filtered_doc)
            except Exception as e:
                print(f"Error filtering document: {e}, skipping")
        
        print(f"Successfully filtered {len(filtered_texts)} documents")
        
        # Create new collection
        print("Creating new vector database collection...")
        vectorstore = Chroma.from_documents(
            documents=filtered_texts,
            embedding=embeddings,
            persist_directory=vectordb_config["location"],
            collection_name=vectordb_config["collection_name"]
        )
        print(f"Vector database created successfully with {len(filtered_texts)} embeddings")
    except Exception as e:
        print(f"Error creating vector database: {e}")
        sys.exit(1)
    
    # Set up hybrid retrieval system for better results
    if config["retrieval"]["use_hybrid_search"]:
        print("Setting up hybrid search retrieval...")
        try:
            # Create keyword-based retriever
            bm25_retriever = BM25Retriever.from_documents(filtered_texts)
            bm25_retriever.k = config["retrieval"]["k"]
            
            # Create vector-based retriever
            vector_retriever = vectorstore.as_retriever(
                search_kwargs={"k": config["retrieval"]["k"]}
            )
            
            # Create ensemble retriever
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.3, 0.7]  # Favor semantic search
            )
            print("Hybrid search initialized")
        except Exception as e:
            print(f"Error setting up hybrid search: {e}, falling back to vector search only")
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": config["retrieval"]["k"]}
            )
    else:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": config["retrieval"]["k"]}
        )
    
    # Initialize the LLM with M2 Pro optimizations
    print("Initializing LLM...")
    model_config = config["model"]
    
    try:
        llm = LlamaCpp(
            model_path=model_config["path"],
            n_ctx=model_config["context_size"],
            n_gpu_layers=model_config["gpu_layers"],
            f16_kv=model_config["f16_kv"],
            use_mlock=model_config.get("use_mlock", True),
            verbose=False
        )
        print(f"LLM initialized with context size {model_config['context_size']}")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)
    
    # Set up conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create a retrieval-based conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )
    
    return qa_chain, llm

def main(query: str):
    """Process a query with better error handling and response formatting."""
    llm = None
    qa_chain = None
    try:
        print("Initializing RAG system optimized for M2 Pro...")
        qa_chain, llm = initialize_qa_system()
        
        print(f"Processing query: {query}")
        # Use the newer invoke() method instead of deprecated __call__()
        response = qa_chain.invoke({"question": query})
        
        print("\n" + "="*50)
        print("Answer:")
        print(response["answer"])
        print("="*50)
        
    except KeyboardInterrupt:
        print("\nOperation canceled by user")
    except Exception as e:
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Enhanced cleanup for LlamaCpp resources
        if 'llm' in locals() and llm is not None:
            try:
                # Use a safer approach to free resources
                import gc
                # Remove references to the model
                if 'qa_chain' in locals() and qa_chain is not None:
                    del qa_chain
                
                # Try to manually clean up the model
                if hasattr(llm, '_model') and llm._model is not None:
                    llm._model = None
                
                del llm
                # Force garbage collection
                gc.collect()
                print("Resources cleaned up")
            except Exception as e:
                print(f"Warning: Error during cleanup (this doesn't affect results): {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query your documents using a local LLM optimized for Apple M2 Pro."
    )
    parser.add_argument(
        "query", 
        type=str, 
        nargs="?",  # Make it optional
        help="The question you want to ask about your documents."
    )
    args = parser.parse_args()
    
    if not args.query:
        query = input("Enter your question: ")
    else:
        query = args.query
    
    # Register signal handlers for graceful shutdown
    import signal
    import atexit
    
    def graceful_exit():
        # Force Python to not run any cleanup that might cause issues
        print("Exiting gracefully...")
        os._exit(0)  # More forceful than sys.exit, skips cleanup routines
    
    # Register the exit handler for different signals
    atexit.register(graceful_exit)
    signal.signal(signal.SIGINT, lambda sig, frame: graceful_exit())
    signal.signal(signal.SIGTERM, lambda sig, frame: graceful_exit())
    
    # Run the main function
    main(query)
    
    # Exit immediately to avoid cleanup errors
    os._exit(0)