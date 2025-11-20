# **Bot-or-Not**: Temporal Graph-Based Bot Detection on Twitter

## Overview

We study bot detection using a subgraph of **TwiBot-22**, the largest real Twitter bot dataset.
Given constraints on computational footprint, the dataset is **sampled** and stored in a compact local DuckDB database.
We then experiment with:

1. Graph-theoretic bot detection
2. BotRGCN
3. Time aware RGCN on DTDG

---
##  Results


---

##  Setup Instructions

### 1. Create `.env`

```env
DATA_ROOT=/absolute/path/to/data
DB_PATH=/absolute/path/to/db/twitter_graph.duckdb
DATASET_ROOT = /absolute/path/to/dataset
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Sample the Graph

```bash

```

### 4. Setup the DuckDB Database
```bash

```

### 4. Build the Pytorch Geometric Dataset
```bash

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
├── model/
├── notebooks/
├── src/
├── scripts/
│
├── .env
├── main.py
├── pyproject.toml
└── README.md
```

### 5. Train the models
```bash

```

### 6. Run Inference
```bash

```
