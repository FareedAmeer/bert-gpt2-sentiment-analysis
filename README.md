# DistilBERT vs DistilGPT-2 Sentiment Analysis

## Project Overview
This project compares two fine-tuned transformer models for binary sentiment analysis of movie reviews.

## Models
1. DistilBERT (Encoder-only)
2. DistilGPT-2 (Decoder-only)

## Dataset
IMDb Movie Reviews Dataset

- Training samples: 2,000
- Testing samples: 500

## Evaluation Results

| Metric | DistilBERT | DistilGPT-2 |
|---|---:|---:|
| Accuracy | 86.00% | 86.80% |
| Precision | 84.92% | 86.59% |
| Recall | 86.99% | 86.59% |
| F1 Score | 85.94% | 86.59% |

## Technologies
- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Hub
- Streamlit
- GitHub

## Model Repositories

- DistilBERT: https://huggingface.co/YOUR_USERNAME/distilbert-sentiment
- DistilGPT-2: https://huggingface.co/YOUR_USERNAME/distilgpt2-sentiment

Replace YOUR_USERNAME with your Hugging Face username.

## Application
A Streamlit web application will allow users to select a model, enter a movie review, and view sentiment predictions.

## Author
BSIT Student Project
