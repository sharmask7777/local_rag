# Local LLM-based RAG System (Optimized for M2 Pro Mac)

This project implements a Retrieval-Augmented Generation (RAG) system using a locally run Large Language Model (LLM) on macOS, specifically optimized for M2 Pro with 16GB RAM. It processes a directory of documents, creates embeddings, and allows for question-answering based on the document content.

## 🚀 Latest Improvements

This version includes several optimizations for M2 Pro Macs:
- Better embedding model (BAAI/bge-small-en-v1.5) with MPS acceleration
- Qdrant vector database for persistent storage and better retrieval
- Hybrid search combining keyword and semantic search
- Conversation memory for follow-up questions
- Optimized LLM parameters for Apple Silicon
- A configuration system for easy customization

## 📋 Prerequisites

- macOS (optimized for Apple Silicon)
- Python 3.9+
- Homebrew (for system dependencies)

## 🔧 Quick Setup

1. **Install dependencies**:
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate
   
   # Install Python packages
   pip install -r requirements.txt
   
   # Install system dependencies (if not already done)
   brew install libmagic poppler tesseract
   ```

2. **Download a model**:
   ```bash
   # Download Llama 2 7B (recommended for M2 Pro with 16GB RAM)
   python download_model.py --model llama2
   
   # For higher accuracy (requires more RAM):
   # python download_model.py --model llama3  # This downloads Llama 2 13B
   
   # Alternative model:
   # python download_model.py --model mistral  # Mistral 7B
   ```

3. **Prepare your documents**:
   Place your PDF documents in the `documents` directory.

4. **Run the system**:
   ```bash
   python rag_script.py "What do my documents say about..."
   ```

## ⚙️ Configuration

The system uses a `config.yaml` file for customization. Key settings:

```yaml
model:
  path: ./models/llama-2-7b-chat.Q4_K_M.gguf  # Model path
  context_size: 4096  # Context window size (adjust based on RAM)
  
embedding:
  model_name: BAAI/bge-small-en-v1.5  # Embedding model
  use_gpu: true  # Use MPS acceleration

documents:
  chunk_size: 500  # Document chunk size
  chunk_overlap: 100  # Overlap between chunks

retrieval:
  k: 4  # Number of documents to retrieve
  use_hybrid_search: true  # Use both keyword and semantic search
```

## 📁 Project Structure

```
local_rag/
│
├── venv/                      # Virtual environment
├── models/                    # LLM models 
│   └── llama-2-7b-chat.Q4_K_M.gguf
├── documents/                 # Your documents
│   └── (your PDFs)
├── qdrant_db/                 # Vector database storage
├── config.yaml                # Configuration file
├── rag_script.py              # Main RAG script
├── download_model.py          # Model downloading utility
└── requirements.txt           # Python dependencies
```

## 🔍 Usage Examples

1. **Basic question**:
   ```bash
   python rag_script.py "What are the key concepts in this document?"
   ```

2. **Interactive mode**:
   ```bash
   python rag_script.py
   # Then enter your question when prompted
   ```

3. **Download a different model**:
   ```bash
   python download_model.py --model mistral
   ```

## 🔧 Troubleshooting

- **Out of memory errors**: Reduce `context_size` in config.yaml to 2048 or 1024
- **Slow performance**: Ensure MPS acceleration is working with `use_gpu: true`
- **Document loading issues**: Check file format compatibility and system dependencies

## 📝 Notes

- The performance depends on your Mac's specifications. The configuration is optimized for M2 Pro with 16GB RAM.
- For better performance with larger documents, consider adjusting chunk size and overlap.
- The system maintains conversation history for follow-up questions.

## Prerequisites

- macOS
- Homebrew
- Python 3.x
- Xcode Command Line Tools

# To run

```
python rag_script.py "what do my notes say about contract testing?"
```

## Installation

1. Install Xcode Command Line Tools:
   ```
   xcode-select --install
   ```

2. Install Homebrew (if not already installed):
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. Install Python (if not already installed):
   ```
   brew install python
   ```

4. Create and navigate to the project directory:
   ```
   mkdir llm_project
   cd llm_project
   ```

5. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

6. Install required packages:
   ```
   pip install langchain chromadb sentence-transformers llama-cpp-python pypdf unstructured
   ```

7. Install additional system dependencies:
   ```
   brew install libmagic poppler tesseract
   ```

8. Add Homebrew to PATH (add to ~/.zshrc or ~/.bash_profile):
   ```
   export PATH="/opt/homebrew/bin:$PATH"
   ```
   Run `source ~/.zshrc` or `source ~/.bash_profile` after adding.

9. Download a GGUF model:
   ```
   mkdir models
   cd models
   wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
   cd ..
   ```

## Project Structure

```
llm_project/
│
├── venv/
├── models/
│   └── llama-2-7b-chat.Q4_K_M.gguf
├── documents/
│   └── (your PDF documents here)
└── rag_script.py
```

## Usage

1. Place your PDF documents in the `documents` directory.

2. Run the script:
   ```
   python rag_script.py
   ```

3. The script will process the documents, create embeddings, and set up a question-answering system.

4. You can modify the query in the script to ask different questions about your documents.

## Troubleshooting

### libmagic error
If you encounter `ImportError: failed to find libmagic`, try:
```
brew install libmagic
pip uninstall python-magic
pip install python-magic
```

### PDFInfoNotInstalledError
Ensure Poppler is installed and in your PATH:
```
brew install poppler
which pdfinfo  # Should return a path
```

### TesseractNotFoundError
Ensure Tesseract is installed and in your PATH:
```
brew install tesseract
which tesseract  # Should return a path
```

You may need to set the Tesseract path in your script:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
```

## Redownloading Dependencies

To reinstall all dependencies:

1. Generate a requirements file:
   ```
   pip freeze > requirements.txt
   ```

2. Remove all packages:
   ```
   pip uninstall -y -r <(pip freeze)
   ```

3. Reinstall packages:
   ```
   pip install -r requirements.txt
   ```

## Notes

- The performance of the local LLM depends on your Mac's specifications.
- Adjust the model size or settings if you encounter performance issues.
- GGUF (GPT-Generated Unified Format) files are optimized for running LLMs on consumer hardware.

