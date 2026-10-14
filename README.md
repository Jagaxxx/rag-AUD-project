# Retrieval-Augmented Generation (RAG) System over a Document Set

Course: **Analysis of Unstructured Data** 


## Repository Structure

```text
rag-AUD-project/
│
├── configs/
│   └── config.yaml                     # configuration file for hyperparameters (chunk size, overlap, top-k, model names)
│
├── data/
│   ├── interim/                        # Intermediate text data after cleaning and pre-processing
│   ├── processed/                      # Processed features, chunk metadata, and serialized data
│   └── raw/                            # Original document corpus (PDFs, text files, markdown) — excluded from Git
│
├── models/                             # Saved FAISS index files, vector databases, and local model weights — excluded from Git
│
├── notebooks/                         
│   ├── 01_data_exploration.ipynb      
│   ├── 02_rag_pipeline.ipynb         
│   ├── 03_evaluation_metrics.ipynb   
│   └── 04_interpretability_and_errors.ipynb
│
├── reports/
│   └── figures/                    
│
├── src/                                
│   ├── data.py                         # Functions for loading raw documents and cleaning text
│   ├── features.py                     # Functions for text chunking and generating embeddings via sentence-transformers
│   ├── models.py                       # Functions for managing the FAISS index and interfacing with the LLM reader
│   └── train.py                        # Orchestration script that coordinates data loading, embedding, indexing, and saving
│
├── .gitignore                      
├── README.md                           # Main project documentation, setup guide, and file descriptions
└── requirements.txt              
```