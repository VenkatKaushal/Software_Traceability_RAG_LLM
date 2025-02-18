from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from tqdm.asyncio import tqdm
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KGTableRetriever,
    KnowledgeGraphRAGRetriever
)
import re

from llama_index.core.indices.keyword_table import KeywordTableGPTRetriever

from typing import List, Union
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
import asyncio

from indexing.constants import CLASS_NAME_LABEL
from prompts.templates import SCENARIO_GEN_TEMPLATE


# class GraphRetriever(BaseRetriever):
#     def __init__(self, graph, req_nodes, top_k=5):
#         """
#         graph: nx.Graph representing requirement relationships.
#         req_nodes: list of Document nodes (requirements).
#         top_k: number of neighbors to consider per matched node.
#         """
       
#         self.req_nodes = {node.metadata["file_name"]: node for node in req_nodes}
#         self.graph = graph
#         self.top_k = top_k

#     def _retrieve(self, query: str):
        
#         matched = []
#         for file_name, node in self.req_nodes.items():
#             if query.lower() in node.text.lower():
#                 matched.append(file_name)

       
#         expanded = set(matched)
#         for file_name in matched:
            
#             neighbors = list(self.graph.neighbors(file_name))
#             expanded.update(neighbors[:self.top_k])
        
#         return [self.req_nodes[name] for name in expanded if name in self.req_nodes]


class GraphRetriever(BaseRetriever):
    def __init__(self, graph_path: str, req_nodes: list[Document], top_k=5):
        """
        Graph-based retriever for requirement-to-code traceability.
        graph_path: Path to stored requirement graphs.
        req_nodes: list of Document nodes.
        top_k: Number of neighbors to include.
        """
        self.req_nodes = {node.metadata["file_name"]: node for node in req_nodes}
        self.graphs = {file: nx.read_gexf(os.path.join(graph_path, file)) for file in os.listdir(graph_path)}
        self.top_k = top_k

    def _retrieve(self, query: str):
        """
        Builds a graph that links requirements to code files based on class mentions.
        """
        os.makedirs(save_dir, exist_ok=True)
        class_names = extract_class_names_from_code(code_dir)
        
        for node in requirement_nodes:
            graph = nx.DiGraph()
            file_name = node.metadata.get("file_name")
            graph.add_node(file_name, text=node.text)
    
            # Link to matching Java classes
            for class_name in class_names:
                if class_name in node.text:
                    graph.add_edge(file_name, class_name, relation="related_to_class")
    
            # Save the graph
            nx.write_gexf(graph, f"{save_dir}/{file_name}.gexf")
        return graph

    def extract_class_names_from_code(code_dir):
        """
        Extracts Java class names from all Java files in the given directory.
        """
        class_names = set()
        
        for file_name in os.listdir(code_dir):
            if file_name.endswith(".java"):
                with open(os.path.join(code_dir, file_name), 'r', encoding="utf-8") as f:
                    content = f.read()
                    match = re.search(r'class\s+(\w+)', content)  # Find class name
                    if match:
                        class_names.add(match.group(1))  # Add class name
        
        return class_names






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
    # queries = response.text.split("\n")
    pattern = r"Queries:\s*((?:\d*\.\s*|)([^?]+(?:\?|$))+)"
    matches = re.findall(pattern, str(response))

    
    queries = []
    for match in matches:
        extracted_queries = re.findall(r"(?:\d*\.\s*|)([^?]+(?:\?|$))", match[0])  
        queries.extend(extracted_queries)

    for query in queries:
        print(query)
    return queries


def fuse_results(results_dict, similarity_top_k: int = 2):
    """Fuse results."""
    k = 60.0 
    fused_scores = {}
    text_to_node = {}

    
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

    
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    
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
