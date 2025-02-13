from concurrent.futures import ThreadPoolExecutor
import networkx as nx
# import QueryBundle
from llama_index.core import QueryBundle

# import NodeWithScore
from llama_index.core.schema import NodeWithScore
from tqdm.asyncio import tqdm

# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KGTableRetriever,
    KnowledgeGraphRAGRetriever
)

from llama_index.core.indices.keyword_table import KeywordTableGPTRetriever

from typing import List, Union
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
import asyncio

from indexing.constants import CLASS_NAME_LABEL
from prompts.templates import SCENARIO_GEN_TEMPLATE


class FusionRetriever(BaseRetriever):
    """Ensemble retriever with fusion."""

    def __init__(
        self,
        llm,
        retrievers: List[BaseRetriever],
        query_gen_prompt: str = SCENARIO_GEN_TEMPLATE,
        similarity_top_k: int = 2,
        num_queries: int = 4,
    ) -> None:
        """Init params."""
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        self._llm = llm
        self._num_queries = num_queries
        self._query_gen_prompt = query_gen_prompt
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        queries = generate_queries(
            self._llm, 
            query_bundle.query_str, 
            self._query_gen_prompt,
            self._num_queries
        )
        
        results = asyncio.run(run_queries(queries, self._retrievers))
        final_results = fuse_results(
            results, similarity_top_k=self._similarity_top_k
        )

        return final_results


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: Union[KGTableRetriever, KeywordTableGPTRetriever, KnowledgeGraphRAGRetriever],
        mode: str = "OR",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        kg_nodes = self._kg_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        kg_ids = {n.node.node_id for n in kg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


def generate_queries(
        llm, 
        query_str: str, 
        query_gen_prompt: str = SCENARIO_GEN_TEMPLATE,
        num_queries: int = 4
    ):
    fmt_prompt = query_gen_prompt.format(
        num_queries=num_queries - 1, query=query_str
    )
    response = llm.complete(fmt_prompt)
    queries = response.text.split("\n")
    return queries


def fuse_results(results_dict, similarity_top_k: int = 2):
    """Fuse results."""
    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    text_to_node = {}

    # compute reciprocal rank scores
    for nodes_with_scores in results_dict.values():
        for rank, node_with_score in enumerate(
            sorted(
                nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True
            )
        ):
            text = node_with_score.node.get_content()
            text_to_node[text] = node_with_score
            if text not in fused_scores:
                fused_scores[text] = 0.0
            fused_scores[text] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes: List[NodeWithScore] = []
    for text, score in reranked_results.items():
        reranked_nodes.append(text_to_node[text])
        reranked_nodes[-1].score = score

    return reranked_nodes[:similarity_top_k]


async def run_queries(queries, retrievers):
    """Run queries against retrievers."""
    tasks = []
    for query in queries:
        for i, retriever in enumerate(retrievers):
            tasks.append(retriever.aretrieve(query))

    task_results = await tqdm.gather(*tasks, desc='Generating queries')

    results_dict = {}
    for i, (query, query_result) in enumerate(zip(queries, task_results)):
        results_dict[(query, i)] = query_result

    return results_dict


def get_reachable_nodes(graph, source, distance):
  reachable_nodes = []
  for node, path_length in nx.single_source_shortest_path_length(graph, source).items():
    if path_length is not None and path_length <= distance:
      reachable_nodes.append(node)
  return reachable_nodes


def retrieve_parallel(
        retriever: BaseRetriever, 
        req_nodes, 
        prompt_template: str,
        num_threads=8,
    ):
    progress_bar = tqdm(total=len(req_nodes), desc="Retrieving Nodes", unit="Requirement")
    futures = list()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for req_node in req_nodes:
            future = executor.submit(
                retriever.retrieve,
                prompt_template.format(requirement=req_node.text)
            )
            futures.append((req_node.metadata["file_name"], future))

        results = list()
        for file_name, future in futures:
            result = future.result()
            class_names = [n.metadata[CLASS_NAME_LABEL] for n in result]
            results.append((file_name, class_names))
            progress_bar.update(1)
    
    progress_bar.close()
    results = dict(results)
    return results