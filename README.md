
# NL2CodeTrace: A Code Traceability and Evaluation Framework

## Overview
NL2CodeTrace is a framework designed to process, index, and evaluate traceability links between natural language requirements and source code. The framework leverages various retrieval and indexing techniques, including vector-based retrieval, keyword table indexing, knowledge graph indexing, and BM25 retrieval. 

## Features
- **Vector Indexing**: Uses embeddings for efficient similarity search.
- **Knowledge Graph Indexing**: Captures relationships between concepts.
- **Keyword Table Indexing**: Extracts and indexes keywords for fast retrieval.
- **BM25 Retrieval**: A text-based retrieval model for ranking documents.
- **Fusion Query Engine**: Combines multiple retrievers for enhanced search accuracy.
- **Evaluation Metrics**: Measures retrieval effectiveness and correctness.

## Dependencies
This framework relies on the following Python libraries:
- `llama_index`
- `chroma`
- `neo4j`
- `pickle`
- `json`
- `os`
- `subprocess`
- `time`
- `typing`

Ensure all dependencies are installed before running the framework.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
### Running the Framework
Execute the main script with:
```bash
python main.py
```
### Configuration
Modify `configs` in `main.py` to include different dataset names and embedding models.
```python
configs = [
    ('iTrust', 'bge_large'),
    ('smos', 'bge_m3'),
    ('eANCI', 'bge_m3')
]
```

### Indexing and Retrieval
- **Vector Index**: Uses embeddings to retrieve similar code or requirements.
- **Knowledge Graph Index**: Stores relationships between code and requirements.
- **Keyword Table Index**: Fast retrieval based on keyword extraction.
- **BM25 Retriever**: Traditional text search model.

### Evaluation
Evaluation is conducted using correctness and retrieval performance metrics:
```python
correctness_results = evaluate_response(
    req_nodes=req_nodes,
    query_engines=query_engines,
    solutions_file=solutions_file_path,
    dataset_name=dataset_name
)
```

### Output
Results are stored in JSON format under the `results/` directory.
```bash
results/{dataset_name}_results.json
```


--------
Retrievers.py

## Overview
This module implements retrieval methods for document search, combining multiple retrieval strategies such as vector search, knowledge graph retrieval, and keyword-based retrieval. It includes an ensemble approach with fusion techniques to improve search accuracy.

## Features

### 1. **FusionRetriever**
- Utilizes multiple retrievers (Vector, Knowledge Graph, BM25, etc.)
- Generates multiple queries using an LLM
- Merges results using a reciprocal rank fusion approach

### 2. **CustomRetriever**
- Performs both **Vector search** and **Knowledge Graph search**
- Supports retrieval in `AND` or `OR` mode to refine or expand search
- Aggregates and ranks results from multiple retrieval methods

### 3. **Query Generation & Fusion**
- Uses an LLM to generate multiple search queries
- Applies reciprocal rank fusion (RRF) to merge results from different retrievers
- Adjusts ranking dynamically to prioritize relevant documents

### 4. **Parallel Retrieval Execution**
- Implements multi-threaded retrieval for efficiency
- Runs retrieval tasks asynchronously for improved performance

### 5. **Batch Query Processing**
- Supports large-scale batch processing of queries
- Uses `ThreadPoolExecutor` for concurrent execution

## Code Example
```python
retriever = FusionRetriever(
    llm=your_llm_model,
    retrievers=[vector_retriever, kg_retriever],
    similarity_top_k=5,
    num_queries=3
)
query_bundle = QueryBundle(query_str="Find relevant code snippets")
results = retriever._retrieve(query_bundle)
```

-----------------------------

# Query Engine and Fusion Retriever Module

## Overview
This module provides a framework for querying documents efficiently using multiple retrieval strategies. It implements fusion-based retrieval, parallel querying, and response synthesis for improved accuracy and performance.

## Features

### 1. **Fusion Query Engine (FusionRetriever)**
- Combines multiple retrievers for improved search accuracy.
- Generates multiple queries dynamically using an LLM.
- Synthesizes responses using summarization techniques.

### 2. **Parallel and Sequential Query Execution**
- **Parallel querying**: Utilizes multi-threading to query multiple documents efficiently.
- **Sequential querying**: Processes queries one at a time for improved stability in resource-limited environments.

### 3. **Custom Query Engine**
- Integrates vector-based retrieval and knowledge graph retrieval.
- Supports `AND` and `OR` retrieval modes for flexible search behavior.
- Uses response synthesizers for refined query results.


### 4. **LLM Support**
- Supports embedding models for text representation.
- Custom LLM implementation (`GemmaLLM`) for text generation and prediction.
- Uses Hugging Face models with `transformers` for fine-tuned LLM operations.

### 5. **Custom LLM Implementation (GemmaLLM)**
- Loads an open-source LLM from Hugging Face.
- Supports chat and completion-based text generation.
- Runs inference on CPU or GPU for optimized performance.

### 6. **Java Static Analysis Tool**
- **AST Generation**: Uses JavaParser to construct an Abstract Syntax Tree (AST).
- **CFG Generation**: Uses Soot to generate a Control Flow Graph (CFG).
- **FDG Extraction**: Uses BCEL to extract a Functional Dependency Graph (FDG).
- Outputs analysis results as a JSON file.
