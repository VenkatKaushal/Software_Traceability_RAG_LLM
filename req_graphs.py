import os
import re
import networkx as nx
import chardet
from llama_index.core.schema import Document
from req2nodes import get_requirements_nodes

def read_file_safely(file_path):
    """Reads a file with automatic encoding detection."""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding'] if result['encoding'] else 'utf-8'
    
    print(f"Reading file {file_path} with detected encoding: {encoding}")
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        return f.read()




def normalize_filename(filename):
    """Normalize filenames by converting to lowercase and removing extensions."""
    normalized = os.path.splitext(filename.lower())[0]
    print(f"Normalized filename: {filename} -> {normalized}")
    return normalized




def get_code_filenames(code_dir):
    """Retrieve all filenames in the code directory."""
    code_filenames = set()
    for root, _, files in os.walk(code_dir):
        for file in files:
            if file.endswith((".java", ".py", ".cpp", ".h", ".txt", ".xml", ".json")):
                normalized_file = normalize_filename(file)
                code_filenames.add(normalized_file)
    print(f"Code filenames detected: {code_filenames}")
    return code_filenames






def build_traceability_graph(requirement_nodes: list[Document], code_dir: str) -> nx.Graph:
    """
    Builds a traceability graph linking requirements to actual code filenames from the code directory.
    """
    graph = nx.DiGraph()
    save_dir = "requirements_graphs"
    os.makedirs(save_dir, exist_ok=True)
    
    filename_pattern = re.compile(r'\b[A-Za-z0-9_]+\b', re.IGNORECASE)
    
    code_filenames = get_code_filenames(code_dir)
    found_filenames = set()
    
    for node in requirement_nodes:
        req_id = node.metadata.get("file_name", f"req_{hash(node.text) % 100000}")
        print(f"Processing requirement: {req_id}")
        graph.add_node(req_id, text=node.text, type="requirement")
        
        
        mentioned_files = filename_pattern.findall(node.text)
        print(f"Raw extracted filenames: {mentioned_files}")
        normalized_files = {normalize_filename(f) for f in mentioned_files}
        print(f"Normalized filenames: {normalized_files}")
        found_filenames.update(normalized_files)
        
        for file in normalized_files:
            if file in code_filenames:
                print(f"Comparing requirement '{req_id}' with code file '{file}'")
                graph.add_node(file, type="code_file")
                graph.add_edge(req_id, file, relation="mentions")
    
    print(f"Total unique filenames found in requirements: {len(found_filenames)}")
    print(f"Saving graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    nx.write_gexf(graph, f"{save_dir}/traceability_graph.gexf")
    return graph





def get_traceability_graph(base_dir: str, code_dir: str, all_req_files_path: str = 'all_req_filenames.txt') -> nx.Graph:
    """
    Retrieves requirement nodes and builds a traceability graph.
    """
    print("Retrieving requirement nodes...")
    nodes = get_requirements_nodes(base_dir, all_req_files_path)
    print(f"Total requirement nodes retrieved: {len(nodes)}")
    return build_traceability_graph(nodes, code_dir)




def main():
    base_dir = "data_repos/ftlr/datasets/eANCI"
    code_dir = "data_repos/ftlr/datasets/eANCI/code"
    print("Starting traceability graph generation...")
    graph = get_traceability_graph(base_dir, code_dir)
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
    print("Graph nodes:", list(graph.nodes)[:10], "...")  
    print("Graph edges:", list(graph.edges)[:10], "...")  



if __name__ == "__main__":
    main()