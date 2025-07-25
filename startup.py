#!/usr/bin/env python3
"""
Startup script for Advanced Keyword Extraction API
This script helps with initial setup and model downloading
"""

import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and log the result"""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8 or higher is required")
        return False
    logger.info(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    logger.info("Installing dependencies...")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    return True

def download_spacy_model():
    """Download SpaCy model"""
    logger.info("Downloading SpaCy model...")
    
    # Download the English model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading SpaCy English model"):
        return False
    
    return True

def download_nltk_data():
    """Download NLTK data"""
    logger.info("Downloading NLTK data...")
    
    nltk_script = """
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
print("NLTK data downloaded successfully")
"""
    
    if not run_command(f'python -c "{nltk_script}"', "Downloading NLTK data"):
        return False
    
    return True

def test_imports():
    """Test if all required modules can be imported"""
    logger.info("Testing imports...")
    
    test_script = """
try:
    import spacy
    import keybert
    import transformers
    import sklearn
    import numpy
    import pandas
    import fastapi
    import uvicorn
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
"""
    
    if not run_command(f'python -c "{test_script}"', "Testing module imports"):
        return False
    
    return True

def test_spacy_model():
    """Test if SpaCy model loads correctly"""
    logger.info("Testing SpaCy model...")
    
    test_script = """
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("This is a test sentence.")
    print("✅ SpaCy model loaded successfully")
except Exception as e:
    print(f"❌ SpaCy model error: {e}")
    exit(1)
"""
    
    if not run_command(f'python -c "{test_script}"', "Testing SpaCy model"):
        return False
    
    return True

def main():
    """Main setup function"""
    logger.info("🚀 Starting Advanced Keyword Extraction API setup...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        logger.error("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Download SpaCy model
    if not download_spacy_model():
        logger.error("❌ Failed to download SpaCy model")
        sys.exit(1)
    
    # Download NLTK data
    if not download_nltk_data():
        logger.error("❌ Failed to download NLTK data")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        logger.error("❌ Failed to import required modules")
        sys.exit(1)
    
    # Test SpaCy model
    if not test_spacy_model():
        logger.error("❌ Failed to load SpaCy model")
        sys.exit(1)
    
    logger.info("🎉 Setup completed successfully!")
    logger.info("You can now run the API with: python -m uvicorn app.main:app --reload")
    logger.info("Or use: python app/main.py")

if __name__ == "__main__":
    main() 