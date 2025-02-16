import os
from llama_index.core import Document
from tqdm.auto import tqdm
import networkx as nx  


class RequirementsNodesCreator:
    def __init__(self, base_dir: str, req_contents: dict[str, str]):
        self.base_dir = base_dir
        self.req_contents = req_contents

    def create_nodes(self):
        docs = list()
        for file_name, content in tqdm(self.req_contents.items(), desc="Creating Requirement Docs"):
            doc = Document(
                text=content,
                metadata={
                    "file_name": file_name
                },
                excluded_embed_metadata_keys=["file_name", "questions_this_excerpt_can_answer"],
                excluded_llm_metadata_keys=["file_name", "questions_this_excerpt_can_answer"]
            )
            docs += [doc]
        return docs


def get_requirements_docs(
        base_dir: str,
        all_req_files_path: str = 'all_req_filenames.txt'
    ):
    all_req_files_path = os.path.join(base_dir, all_req_files_path)
    all_req_files = [f_name.strip() for f_name in open(all_req_files_path)]

    all_req_contents = dict()
    for f_name in tqdm(all_req_files):
        with open(os.path.join(base_dir, 'req', f_name)) as f:
            all_req_contents[f_name] = f.read()

    return all_req_contents


def get_requirements_nodes(
        base_dir: str,
        all_req_files_path: str = 'all_req_filenames.txt'
    ):
    all_req_contents = get_requirements_docs(base_dir, all_req_files_path)
    req_parser = RequirementsNodesCreator(base_dir, all_req_contents)
    req_nodes = req_parser.create_nodes()
    return req_nodes




def build_requirements_graph(requirement_nodes: list[Document]) -> nx.Graph:
    """
    Builds a graph from requirement nodes.
    Each node represents a requirement document (keyed by its file name).
    An edge is added from one requirement to another if the text of one requirement
    contains the other’s file name.
    """
    graph = nx.Graph()
    
   
    for node in requirement_nodes:
        file_name = node.metadata.get("file_name")
        graph.add_node(file_name, text=node.text)
    
  
    for node in requirement_nodes:
        source = node.metadata.get("file_name")
        for other_node in requirement_nodes:
            target = other_node.metadata.get("file_name")
            if source != target and target in node.text:
                graph.add_edge(source, target)
    
    return graph


def get_requirements_graph(
        base_dir: str,
        all_req_files_path: str = 'all_req_filenames.txt'
    ) -> nx.Graph:
    """
    Retrieves requirement nodes and builds a requirements graph.
    """
    nodes = get_requirements_nodes(base_dir, all_req_files_path)
    return build_requirements_graph(nodes)