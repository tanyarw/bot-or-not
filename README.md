# **Bot-or-Not**: Temporal Graph-Based Bot Detection on Twitter

## Overview

We study bot detection using a subgraph of **TwiBot-22**, the largest real Twitter bot dataset.
Given constraints on computational footprint, the dataset is **sampled** and stored in a compact local DuckDB database.
We then experiment with:

1. Graph-theoretic analysis
2. BotRGCN
3. Time aware RGCN on DTDG

---
##  Results

| Model              | Val Acc (%) | Val F1-macro (%) | Test Acc (%) | Test F1-macro (%) |
|-------------------|-------------|------------------|--------------|--------------------|
| **BotRGCN (Static)** | 81.35       | 66.84            | 58.68        | 58.39             |
| **EvoRGCN (DTDG)**   | **84.81**   | **70.52**        | **61.87**    | **60.26**         |


---

##  Setup Instructions

### 1. Create `.env`

```env
DATA_ROOT=</absolute/path/to>/data
DB_PATH=</absolute/path/to>/db/twitter_graph.duckdb
DATASET_ROOT =</absolute/path/to>/dataset
BOTRGCN_ROOT =</absolute/path/to>/botrgcn
TEST_NODES_PATH="./dataset>/test_user_nodes.pt"
LOG_ROOT = ./logs
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Sample the Graph

Prerequisite: Access to the original [Twibot-22 dataset](https://github.com/LuoUndergradXJTU/TwiBot-22)
```bash
python scripts/sampling.py
```

### 4. Setup the DuckDB Database
```bash
python scripts/db_setup.py
```

### 5. Build the Discrete-Time Dynamic Graph "Snapshots" (DTDG)
```bash
python scripts/processing.py
```

### 5. Build the Node Embeddings and the Pytorch Geometric Dataset for the static graph
Settings for this run are controlled by the file `config.yaml`.
```bash
python scripts/dataset.py
```

### 6. Train the  BotRGCN model
```bash
python scripts/train_static_graph.py
```

### 6. Build the Snapshots Dataset and train the EvoRGCN model
```bash
python src/train.py
```

**Expected Directory Structure**

```
bot-or-not/
│
├── db/
│   └── twitter_graph.duckdb
│
├── data/
│   └── static/
│       ├── sampled_users.jsonl
│       ├── sampled_tweets.jsonl
│       ├── sampled_edges.csv
│       └── bot_labels.csv
├── dataset/
│   └── static/
│       ├── ...
│       ├── node_embeddings_128.pt
│       └── labels.pt
│
├── data_analysis/
├── model/
├── src/
├── scripts/
│
├── .env
├── main.py
├── pyproject.toml
└── README.md
```
