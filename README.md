<<<<<<< HEAD
<<<<<<< HEAD
Week 4 LLM Assessment
Project Overview

This project demonstrates four important areas of Large Language Models.

Task 1: Models, Tokens and Cost
GPT, Claude, Llama and Gemini comparison
Tokenization using Tiktoken
Five different text samples
Words per token
Model cost estimation
8K and 128K context comparison
Model recommendations
Task 2: Capabilities and Limitations
Summarization
Classification
Information extraction
Translation
Code generation
Knowledge cutoff testing
Hallucination testing
Context retrieval
Arithmetic and logic
Business application safeguards
Task 3: Transformer Concepts
NumPy self-attention
Multi-head attention
Output projection
Sinusoidal positional encoding
BERT fill-mask
GPT-2 text generation
Encoder-only vs decoder-only models
Task 4: Prompt Engineering
Golden ticket dataset
Three prompt versions
Classification accuracy
Reasoning experiments
Temperature and top-p experiments
Resume information extraction
JSON mode
Pydantic validation
Prompt improvement
Technologies
Python
Google Gemini API
NumPy
Pandas
Tiktoken
Hugging Face Transformers
PyTorch
Matplotlib
Pydantic
Installation

Create a virtual environment:

bash
python -m venv venv


Activate it:

powershell
venv\Scripts\Activate.ps1


Install dependencies:

bash
pip install -r requirements.txt

Configure API

Create a .env file locally:

env
GEMINI_API_KEY=your_new_api_key
GEMINI_MODEL=your_model_name
RUN_EXTENDED_EXPERIMENTS=true


Never commit .env to GitHub.

Run Tasks
bash
python models_tokens_cost.py
python capabilities.py
python transformer.py
python prompt_toolkit.py

Outputs

The outputs directory contains generated CSV, JSON and PNG assessment results.

Important

API results depend on API availability and account quota.

Unavailable API experiments are recorded as unavailable instead of generating fake results.

Local Transformer
=======
# Week4_LLM_Assessment
A Python project covering LLM fundamentals, tokenization, cost estimation, model capabilities, limitations, transformer architecture, self-attention, positional encoding, and prompt engineering using NumPy, Tiktoken, Hugging Face, OpenAI API, and Pydantic.
>>>>>>> 40465e00ab1f0aadff176d354af87337dd3efdd2
=======
# Week4_LLM_Assessment
A Python project covering LLM fundamentals, tokenization, cost estimation, model capabilities, limitations, transformer architecture, self-attention, positional encoding, and prompt engineering using NumPy, Tiktoken, Hugging Face, OpenAI API, and Pydantic.
>>>>>>> e787a1e19e2cc4f2152f1bd3b3a3a478dc371e3b
