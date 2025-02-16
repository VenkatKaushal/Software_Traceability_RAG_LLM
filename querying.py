
"""
This script defines functions for querying documents in parallel and sequentially using custom query engines and retrievers. It also includes a function to create a fusion query engine that combines results from multiple retrievers. These queries can be executed in parallel using multiple threads to improve efficiency.

Functions:

1. get_fusion_qe:
   - Creates a Fusion Query Engine by initializing a `FusionRetriever` with a list of retrievers and additional parameters. It then creates a response synthesizer using `get_response_synthesizer` and returns a `RetrieverQueryEngine` that utilizes both the retriever and the response synthesizer.

2. query_parallel:
   - Executes queries in parallel using a `ThreadPoolExecutor` with multiple threads. It takes a query engine, a query template, a list of requirement nodes, and the number of threads to use. It returns a dictionary of results mapped to the respective file names.

3. query_sequential:
   - Executes queries sequentially (one after another) for each requirement node. It returns the results in a dictionary, mapping the file names to the query results.

4. custom_query_engine:
   - Creates a custom query engine using a `CustomRetriever` that combines a vector retriever and a knowledge graph retriever. It returns a `RetrieverQueryEngine` with a response synthesizer.

Dependencies:
- concurrent.futures: For parallel query execution using `ThreadPoolExecutor`.
- llama_index: Provides classes for document nodes, query engines, and retrievers.
- tqdm.auto: For progress tracking during querying.
- tqdm.asyncio: For asynchronous progress bars.
"""


from concurrent.futures import ThreadPoolExecutor
from llama_index.core import Document, Settings
from tqdm.auto import tqdm
from llama_index.core.response_synthesizers import get_response_synthesizer
from typing import List, Union
from llama_index.core.query_engine import BaseQueryEngine
from retrievers import CustomRetriever
from typing import List, Dict, Any
from tqdm.asyncio import tqdm
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KGTableRetriever,
    KnowledgeGraphRAGRetriever,
)
from llama_index.core.indices.keyword_table import KeywordTableGPTRetriever

from typing import List
from prompts.templates import SCENARIO_GEN_TEMPLATE
from retrievers import FusionRetriever


def get_fusion_qe(
        retrievers,
        query_gen_prompt=SCENARIO_GEN_TEMPLATE,
        similarity_top_k = 10,
        num_queries = 4,
        summarize_mode = 'tree_summarize'
):
    fusion_retriever = FusionRetriever(
        llm=Settings.llm,
        query_gen_prompt=query_gen_prompt,
        retrievers=retrievers, 
        similarity_top_k=similarity_top_k,
        num_queries=num_queries
    )
    response_synthesizer = get_response_synthesizer(
        llm=Settings.llm,
        response_mode=summarize_mode,
        use_async=True
     )
    fusion_qe = RetrieverQueryEngine(retriever=fusion_retriever, response_synthesizer=response_synthesizer)
    return fusion_qe



"""
Use this version of query_parallel for even best quering results. This queries iteratively by slowing increasing the information to the LLM.
"""

def query_parallel(
        query_engine: BaseQueryEngine,
        query_template: str, 
        req_nodes: List[Document], 
        num_threads=8,
    ):
    progress_bar = tqdm(total=len(req_nodes), desc="Querying Requirements", unit="Requirement")
    futures = list()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for req_node in req_nodes:
            future = executor.submit(
                query_engine.query,
                query_template.format(requirement=req_node.text)
            )
            futures.append((req_node.metadata["file_name"], future))

        results = list()
        for file_name, future in futures:
            results.append((file_name, future.result()))
            progress_bar.update(1)
    print("Parrallel processes completed")
    progress_bar.close()
    results = dict(results)
    return results



    

def query_sequential(
        query_engine: BaseQueryEngine,
        query_template: str, 
        req_nodes: List[Document]
    ) -> Dict[str, Any]:
    """
    Runs the query sequentially instead of using multiple threads.

    Args:
        query_engine (BaseQueryEngine): The query engine to execute queries.
        query_template (str): The template string for formatting queries.
        req_nodes (List[Document]): The list of document nodes to process.

    Returns:
        Dict[str, Any]: A dictionary mapping file names to query results.
    """
    results = {}
    progress_bar = tqdm(total=len(req_nodes), desc="Querying Requirements", unit="Requirement")

    for req_node in req_nodes:
        file_name = req_node.metadata["file_name"]
        query = query_template.format(requirement=req_node.text)
        results[file_name] = query_engine.query(query)
        progress_bar.update(1)

    progress_bar.close()
    return results




def custom_query_engine(
        vector_retriever: VectorIndexRetriever,
        kg_retriever: Union[KGTableRetriever, KeywordTableGPTRetriever, KnowledgeGraphRAGRetriever],
        mode: str = "OR"
):
    retriever = CustomRetriever(
        vector_retriever, 
        kg_retriever,
        mode=mode
    )

    response_synthesizer = get_response_synthesizer()
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    return query_engine