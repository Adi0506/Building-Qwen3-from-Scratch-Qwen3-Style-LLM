# Building Qwen3 from Scratch

[![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-F00000?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/) 
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FF6C37?style=for-the-badge&logo=HuggingFace&logoColor=white)](https://huggingface.co/) 
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/) 
[![Transformers](https://img.shields.io/badge/Transformers-000000?style=for-the-badge&logo=transformers&logoColor=white)](https://huggingface.co/docs/transformers/index) 
[![AMP](https://img.shields.io/badge/AMP-000000?style=for-the-badge)](https://pytorch.org/docs/stable/amp.html) 
[![Gradient Accumulation](https://img.shields.io/badge/Gradient%20Accumulation-000000?style=for-the-badge)](https://pytorch.org/docs/stable/notes/amp_examples.html) 

I implemented a **Qwen3-style large language model (LLM) from scratch** in Python and Google Colab.  
This project allowed me to gain hands-on experience with modern transformer architectures and advanced optimization techniques.

---

##  What I Did and Learned

- Designed a transformer architecture with **Grouped-Query Attention (GQA)** and **SwiGLU activations**.  
- Applied **Rotary Positional Embeddings (RoPE)** for improved context handling.  
- Integrated **QK-Norm with RMSNorm** for numerical stability.  
- Developed a **Muon optimizer** using Newton-Schulz orthogonalization for faster, more efficient training.  
- Implemented **gradient accumulation** and **Automatic Mixed Precision (AMP)** for large-batch training.  
- Handled datasets efficiently with **Hugging Face Datasets**, including tokenization and caching.  
- Monitored **loss, accuracy, and perplexity** to evaluate model performance.

---

##  How to Run

Open and run the notebook in Google Colab:  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK)

---

## Next step of development
- Train the model on larger datasets for real-world applications.  
- Deploy it as a chatbot, summarizer, or code-generation assistant.  
- Optimize memory usage and speed for full-scale implementation.
