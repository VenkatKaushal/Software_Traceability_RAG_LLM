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
