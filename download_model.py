#!/usr/bin/env python3
"""
Model downloader for local RAG system.
This script downloads a Llama model optimized for M2 Pro Macs.
"""

import os
import sys
import argparse
import requests
import yaml
from tqdm import tqdm

def load_config():
    """Load configuration from config.yaml"""
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    return None

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get the total file size
    total_size = int(response.headers.get('content-length', 0))
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download with progress bar
    progress_bar = tqdm(
        total=total_size, 
        unit='iB', 
        unit_scale=True,
        desc=f"Downloading {os.path.basename(destination)}"
    )
    
    with open(destination, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                progress_bar.update(len(chunk))
                file.write(chunk)
    
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Downloaded size does not match expected size!")
        return False
    
    return True

def download_model(model_type="llama3"):
    """Download a model based on type"""
    models = {
        "llama3": {
            "name": "Llama 2 13B Chat (Q4_K_M)",
            "url": "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf",
            "path": "models/llama-2-13b-chat.Q4_K_M.gguf"
        },
        "llama2": {
            "name": "Llama 2 7B Chat (Q4_K_M)",
            "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf", 
            "path": "models/llama-2-7b-chat.Q4_K_M.gguf"
        },
        "mistral": {
            "name": "Mistral 7B Instruct v0.2 (Q4_K_M)",
            "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "path": "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        }
    }
    
    if model_type not in models:
        print(f"Error: Model type '{model_type}' not recognized. Available models: {', '.join(models.keys())}")
        return False
    
    model = models[model_type]
    print(f"Downloading {model['name']}...")
    
    # Check if model already exists
    if os.path.exists(model["path"]):
        print(f"Model already exists at {model['path']}. Skipping download.")
        return True
    
    # Create model directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download the model
    success = download_file(model["url"], model["path"])
    
    if success:
        print(f"Model downloaded successfully to {model['path']}")
        
        # Update config to use this model
        config = load_config()
        if config:
            config["model"]["path"] = model["path"]
            with open("config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Updated config.yaml to use the downloaded model")
        
        return True
    else:
        print("Failed to download model")
        return False

def main():
    """Main function for the model downloader"""
    parser = argparse.ArgumentParser(description="Download models for local RAG system")
    parser.add_argument(
        "--model", 
        type=str, 
        default="llama3", 
        choices=["llama3", "llama2", "mistral"],
        help="Model to download (default: llama3)"
    )
    args = parser.parse_args()
    
    print("Model Downloader for Local RAG System")
    print("=====================================")
    print("This script will download a model optimized for M2 Pro Mac")
    
    download_model(args.model)

if __name__ == "__main__":
    main() 