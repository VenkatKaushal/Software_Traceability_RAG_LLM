import pickle
from typing import List
import json
import os
import time
import subprocess


from llama_index import *
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.core.indices import VectorStoreIndex
from evaluation import (
    evaluate_response,
    evaluate_retrieval, 
    get_questions_from_nodes
)
from req2nodes import get_requirements_nodes
from querying import custom_query_engine
from api_models import  set_llm_and_embed
from indexing.utils import get_parser
from indexing.indices import (
    create_keyword_table_index, 
    create_knowledge_graph_index, 
    get_graph_store
)


from querying import get_fusion_qe
from code2graph import get_code_graph_nodes
from indexing.code_index import add_method_call_links_to_docs, create_code_graph_nodes_index
from indexing.utils import get_vector_storage_context
from indexing.indices import create_vector_index
from llama_index.core.schema import Document
from constants import (
    LLMsMap,
    EmbeddingModelsMap,
)
from llama_index.core.indices import (
    KnowledgeGraphIndex, 
)

from parameters import parse_args


def get_vector_index(docs, indices_path, dataset_name):
    print(f"Creating Vector Index...")
    storage_context, _ = get_vector_storage_context(
        chroma_db_path=f'{indices_path}/VI',
        collection_name=f'{dataset_name}',
    )

    indexed_nodes, vector_index = create_vector_index(
        nodes=docs, 
        storage_context=storage_context, 
        num_threads=2,
        save_path=f"{indices_path}/{dataset_name}",
        show_progress=True
    )
    print(f"Vector Index created.")
    return indexed_nodes, vector_index


def get_kg_links_index(
        docs, 
        graph_nodes, 
        dataset_name,
        host_ip='localhost',
        include_embeddings=True,
    ):
    graph_storage_context = get_graph_store(
        # db_name=f'nl2code{dataset_name}',
        db_name=f'neo4j',
        host_ip=host_ip,
    )
    index = create_knowledge_graph_index(
        docs, 
        graph_nodes, 
        graph_storage_context,
        include_embeddings=include_embeddings
    )
    return index


def clean_text(text):
    import re
    """Cleans the requirement text by removing extra spaces, newlines, and special characters."""
    text = text.strip()  # Remove leading/trailing spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text

def get_kw_table_index(
        docs, 
        indices_path, 
        dataset_name,
        use_vector_store=False,
    ) -> VectorStoreIndex:
    print(f"Creating Keyword Table Index...")
    storage_context, vector_store = get_vector_storage_context(
        chroma_db_path=f'{indices_path}/KW',
        collection_name=f'{dataset_name}',
    )
    if use_vector_store:
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
        )
    else:
        _, vector_index = create_keyword_table_index(
            docs=docs, 
            storage_context=storage_context, 
            num_keywords=20,
            num_threads=2,
            save_path=f'{indices_path}/{dataset_name}',
            show_progress=True
        )
    
    print(f"Keyword Table Index created.")
    return vector_index


def get_kg_index(
        docs: List[TextNode],
        dataset_name,
        host_ip='localhost',
    ) -> KnowledgeGraphIndex:

    graph_storage_index = get_graph_store(
        # db_name=f'nl2code{dataset_name}',
        db_name=f'neo4j',
        host_ip=host_ip,
    )
    
    docs = [Document(text=doc.get_content(MetadataMode.LLM)) for doc in docs]

    kg_index = KnowledgeGraphIndex.from_documents(
        documents=docs,
        storage_context=graph_storage_index,
        show_progress=True
    )
    return kg_index


def get_code_nodes(args, dataset_name):
    os.makedirs('results', exist_ok=True)
    args.dataset_name = dataset_name

    base_dir = args.base_dir
    dataset_name = args.dataset_name
    dataset_dir = f'{base_dir}/{dataset_name}'
    indices_path = f"{args.chroma_db_dir}/{dataset_name}"
    os.makedirs(indices_path, exist_ok=True)

    call_graph_file = f'{dataset_name.lower()}_{args.call_graph_file}'


    graph_nodes = get_code_graph_nodes(
        base_dir=dataset_dir,
        all_code_files_path=args.all_code_files_path,
        callgraph_file_name=call_graph_file,
    )

    print(f"Indexing code nodes...")
    code_nodes = create_code_graph_nodes_index(
        graph_nodes=graph_nodes,
        add_method_calls=args.add_method_calls
    )
    print(f"Number of code nodes: {len(code_nodes)}")
    print(f"Adding method call links to docs...")
    return code_nodes, graph_nodes


def execute_dataset(args, dataset_name):

    os.makedirs('results', exist_ok=True)
    args.dataset_name = dataset_name

    base_dir = args.base_dir
    dataset_name = args.dataset_name
    dataset_dir = f'{base_dir}/{dataset_name}'
    indices_path = f"{args.chroma_db_dir}/{dataset_name}"
    os.makedirs(indices_path, exist_ok=True)
    
    solutions_file = args.solutions_file
    solutions_file_path = f'{dataset_dir}/{dataset_name.lower()}_{solutions_file}'

    code_nodes, graph_nodes = get_code_nodes(args, dataset_name)

    #store code_nodes, graph_nodes
    # with open(f'indices/{dataset_name}_code_nodes.pkl', 'wb') as f:
    #     pickle.dump(code_nodes, f)
    # with open(f'indices/{dataset_name}_graph_nodes.pkl', 'wb') as f:
    #     pickle.dump(graph_nodes, f)
    
    add_method_call_links_to_docs(code_nodes, rel_type='parent')
    # with open(f'indices/{dataset_name}_code_nodes_updated.pkl', 'wb') as f:
    #     pickle.dump(code_nodes, f)

    parser = get_parser()
    print(f"Getting nodes from documents...")
    docs = parser.get_nodes_from_documents(code_nodes)
    with open(f'indices/{dataset_name}_docs.pkl', 'wb') as f:
        pickle.dump(docs, f)
    print(f"Number of nodes: {len(docs)}")


    #  Cleaning swap memory before proceeding
    # subprocess.run(["sudo", "swapoff", "-a"])  # Turn off swap
    # subprocess.run(["sudo", "swapon", "-a"])  # Turn swap back on
    time.sleep(5)  # Allow CPU to rest

    indexed_nodes, vector_index = get_vector_index(docs, indices_path, dataset_name)

    # with open(f'indices/{dataset_name}{"_mc" if args.add_method_calls else ""}.pkl', 'wb') as f:
    #     pickle.dump(indexed_nodes, f)
    
    # with open(f'indices/{dataset_name}_vector_index.pkl', 'wb') as f:
    #     pickle.dump(vector_index, f)

    bm25retriever = BM25Retriever(nodes=docs)

    fusion_qe = get_fusion_qe(
        retrievers=[bm25retriever],
        similarity_top_k=args.similarity_top_k,
        num_queries=args.num_queries,
    )

     # Cleaning swap memory before proceeding
    # subprocess.run(["sudo", "swapoff", "-a"])  # Turn off swap
    # subprocess.run(["sudo", "swapon", "-a"])  # Turn swap back on
    print("CPU Sleeping")
    time.sleep(5)  # Allow CPU to rest

    print(f"Creating KG Links Index...")



    kg_links_index = get_kg_links_index(
        docs, 
        graph_nodes,
        dataset_name,
        args.host_ip,
        include_embeddings=args.include_embeddings
    )

    # with open(f'indices/{dataset_name}_kg_links_index.pkl', 'wb') as f:
    #     pickle.dump(kg_links_index, f)

    print(f"KG Links Index created.")

    # print(f"Creating KG Index...")
    # kg_index = get_kg_index(docs, dataset_name, args.host_ip)
    # with open(f'indices/{dataset_name}_kg_index.pkl', 'wb') as f:
    #     pickle.dump(kg_index, f)
    # print("One Done")
    # kw_index = get_kw_table_index(docs, indices_path, dataset_name)
    # with open(f'indices/{dataset_name}_kw_index.pkl', 'wb') as f:
    #     pickle.dump(kw_index, f)
    # print(f"Indices created.")

    #Storage
    




     # Cleaning swap memory before proceeding
    # subprocess.run(["sudo", "swapoff", "-a"])  # Turn off swap
    # subprocess.run(["sudo", "swapon", "-a"])  # Turn swap back on
    time.sleep(5)  # Allow CPU to rest

    query_engines = {
        'vector_index': vector_index.as_query_engine(response_mode='refine', similarity_top_k=10),
        'fusion_qe_index': fusion_qe,
        'kg_links_index': kg_links_index.as_query_engine(),
        # 'kg_index': kg_index.as_query_engine(),
        # 'kw_index': kw_index.as_query_engine(),
        'vector_kg_links': custom_query_engine(
            vector_index.as_retriever(),
            kg_links_index.as_retriever(),
        ),
        # 'vector_kw_table': custom_query_engine(
        #     vector_index.as_retriever(),
        #     # kg_links_index.as_retriever(),
        #     kw_index.as_retriever(),
        # ),
        # 'vector_kg': custom_query_engine(
        #     vector_index.as_retriever(),
        #     # kg_links_index.as_retriever(),
        #     kg_index.as_retriever(),
        # ),
    }

    # with open(f'indices/{dataset_name}_query_engines.pkl', 'wb') as f:
    #     pickle.dump(query_engines, f)

    # Cleaning swap before processing next heavy task
    # subprocess.run(["sudo", "swapoff", "-a"])  # Turn off swap
    # subprocess.run(["sudo", "swapon", "-a"])  # Turn swap back on
    time.sleep(5)  # Allow CPU to rest


    req_nodes = get_requirements_nodes(
        dataset_dir,
        all_req_files_path=args.all_req_filenames,
    )

    # with open(f'indices/{dataset_name}_req_nodes.pkl', 'wb') as f:
    #     pickle.dump(req_nodes, f)

    req_nodes = [
    Document(
        text=clean_text(node.text),  # Clean only the text
        metadata=node.metadata        # Preserve metadata (e.g., file_name)
    )
    for node in req_nodes
]
    
    # with open(f'indices/{dataset_name}_req_nodes_updated.pkl', 'wb') as f:
    #     pickle.dump(req_nodes, f)
    
    print("Evaluating correctness...")
    correctness_results = evaluate_response(
        req_nodes=req_nodes,
        query_engines=query_engines,
        solutions_file=solutions_file_path,
        dataset_name=dataset_name
    )

    # with open(f'indices/{dataset_name}_correctness_results.pkl', 'wb') as f:
    #     pickle.dump(correctness_results, f)

    print("Completed Evaluating")
    questions = get_questions_from_nodes(code_nodes)
    # with open(f'indices/{dataset_name}_questions.pkl', 'wb') as f:
    #     pickle.dump(questions, f)
    print(f"Number of questions: {len(questions)}")
    eval_results = evaluate_retrieval(
        questions, query_engines, dataset_name
    )
    # with open(f'indices/{dataset_name}_eval_results.pkl', 'wb') as f:
    #     pickle.dump(eval_results, f)
    print("Results on the way")
    results = {
        'correctness': correctness_results,
        'evaluation': eval_results,
    }
    with open(f'results/{dataset_name}_results.json', 'w') as f:
        json.dump(results, f, indent=4)


def main():
    configs = [
        # ('iTrust', 'bge_large'),
        # ('smos', 'bge_m3'),
        ('eANCI', 'bge_m3')
        # ('eTour', 'bge_large')
    ]
    args = parse_args()
    dataset_name = 'iTrust'
    embed_model = 'bge_large'


    for i, (dataset_name, embed_model) in enumerate(configs):
        
        llm_name = LLMsMap[args.llm]
        embed_model_name = EmbeddingModelsMap[embed_model]
        print(f"Using LLM: {llm_name}")
        print(f"Using Embedding Model: {embed_model_name}")

        set_llm_and_embed(
            llm_type=args.llm_type,
            llm_name=llm_name,
            embed_model_name=embed_model_name,
        )

        execute_dataset(args, dataset_name)
        
        


if __name__ == '__main__':
    main()