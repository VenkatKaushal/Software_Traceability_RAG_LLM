from argparse import ArgumentParser
from constants import (
    EmbeddingModels,
    LLMTypes, 
    LLMs,
)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='data_repos/ftlr/datasets')
    parser.add_argument('--dataset_name', type=str, default='eTour')
    parser.add_argument('--all_code_files_path', type=str, default='all_code_filenames.txt')
    parser.add_argument('--all_req_filenames', type=str, default='all_req_filenames.txt')
    parser.add_argument('--call_graph_file', type=str, default='method_callgraph.json')
    parser.add_argument('--solutions_file', type=str, default='solution_links_english.txt')
    parser.add_argument('--results_dir', type=str, default='results')

    parser.add_argument('--add_method_calls', action='store_true')
    parser.add_argument('--host_ip', type=str, default='localhost')

    parser.add_argument('--chroma_db_dir', type=str, default='indices')
    parser.add_argument('--collection_name', type=str, default='ftlr_etour')

    parser.add_argument('--num_threads', type=int, default=8)
    parser.add_argument('--save_path', type=str, default='eTour')

    parser.add_argument('--show_progress', type=bool, default=True)
    parser.add_argument('--include_embeddings', type=bool, default=True)
    parser.add_argument('--similarity_top_k', type=int, default=2)
    parser.add_argument('--num_queries', type=int, default=4)
    parser.add_argument('--num_similar_nodes', type=int, default=4)


    parser.add_argument('--embed_model', type=str, choices=list(EmbeddingModels._value2member_map_.keys()), default='bge_m3')
    parser.add_argument('--llm_type', type=str, choices=list(LLMTypes._value2member_map_.keys()), default='anyscale')
    parser.add_argument('--llm', type=str, choices=list(LLMs._value2member_map_.keys()), default='llama3')
    parser.add_argument('--api_key_index', type=int, default=0)
    return parser.parse_args()