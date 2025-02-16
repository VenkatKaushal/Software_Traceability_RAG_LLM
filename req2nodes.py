"""
This script defines the logic for creating and retrieving requirement nodes from files. It loads requirement documents, processes them into nodes, and prepares them for further use in a retrieval-based system.

Classes:

1. RequirementsNodesCreator:
   - A class responsible for creating nodes from the requirement documents. It takes a base directory and a dictionary of requirement file contents, and for each file, it creates a `Document` object with the text and metadata.

Functions:

1. create_nodes:
   - Creates nodes for each requirement document. It iterates through the provided dictionary of requirement contents, creating a `Document` object for each one, with metadata such as the file name. The method returns a list of `Document` nodes.

2. get_requirements_docs:
   - Loads all the requirement files specified in `all_req_filenames.txt` from the base directory and returns their contents in a dictionary, where the keys are the file names and the values are the file contents.

3. get_requirements_nodes:
   - Retrieves all requirement documents and then uses `RequirementsNodesCreator` to create nodes from them. It returns the list of nodes created.

Dependencies:
- os: Used to handle file paths.
- llama_index: Provides the `Document` class for creating documents from text.
- tqdm.auto: For progress tracking during file processing.
"""


import os
from llama_index.core import Document
from tqdm.auto import tqdm


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
