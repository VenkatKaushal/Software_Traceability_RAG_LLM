import os
from llama_index.core import Document
from tqdm.auto import tqdm
import networkx as nx  
import re


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
    Builds a graph that links requirements to code files based on class mentions.
    """
    save_dir = "requirements_graphs"
    code_dir = "data_repos/ftlr/datasets/eANCI/code"
    os.makedirs(save_dir, exist_ok=True)
    class_names = extract_class_names_from_code(code_dir)
    
    for node in requirement_nodes:
        graph = nx.DiGraph()
        file_name = node.metadata.get("file_name")
        graph.add_node(file_name, text=node.text)

        
        for class_name in class_names:
            if class_name in node.text:
                graph.add_edge(file_name, class_name, relation="related_to_class")

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
                match = re.search(r'class\s+(\w+)', content)  
                if match:
                    class_names.add(match.group(1))
    
    return class_names


def get_requirements_graph(
        base_dir: str,
        all_req_files_path: str = 'all_req_filenames.txt'
    ) -> nx.Graph:
    """
    Retrieves requirement nodes and builds a requirements graph.
    """
    nodes = get_requirements_nodes(base_dir, all_req_files_path)
    return build_requirements_graph(nodes)