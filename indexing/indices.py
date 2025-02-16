"""
This script provides various utilities for creating and managing vector and knowledge graph indices using LlamaIndex and Neo4j. It also includes functions for handling embedding and keyword extraction, graph upsertion, and multi-threaded processing of nodes.

Functions and Classes:

1. get_url:
   - Returns the appropriate Neo4j database URL based on the provided host IP. If the host IP is "localhost", it uses the default Neo4j URL.

2. create_vector_index:
   - Creates a vector index by processing nodes through a multi-threaded pipeline, applying embedding transformations, and saving the indexed vector store to the specified path.

3. create_keyword_table_index:
   - Creates a keyword table index by extracting keywords from the provided documents. It uses the `KeywordExtractor` to generate keywords and stores them in a `VectorStoreIndex`.

4. create_knowledge_graph_index:
   - Creates a knowledge graph index by extracting call graph links from provided graph nodes and upserting the relationships into a `KnowledgeGraphIndex`. This function also handles creating triples from class and method relationships.

5. get_graph_store:
   - Creates and returns a Neo4j graph store connection. It checks if the database exists, creates it if necessary, and returns a `StorageContext` configured for the graph store.

6. create_neo4j_database:
   - Checks if the specified Neo4j database exists. If not, it creates the database. This function uses a Neo4j driver to interact with the database and ensure its existence.

Dependencies:
- pickle: For serializing and saving the vector index and other data.
- kuzu: Used for graph storage with a custom KuzuGraphStore.
- llama_index: Provides functions and classes for node processing and indexing.
- neo4j: For interacting with the Neo4j graph database.
- tqdm.auto: For displaying progress bars during processing.
- constants: Contains labels for knowledge graph and vector indices.
"""


import logging
import os
import pickle
from llama_index.core.indices import KnowledgeGraphIndex, KeywordTableIndex
from prompts.templates import (
    CODE_KEYWORD_EXTRACT_TEMPLATE_TMPL
)

from indexing.utils import run_pipeline_multithreaded
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from tqdm.auto import tqdm
from code2graph import  extract_call_graph_links
from code2graph import (
    CLASS_NAME_LABEL,
    METHOD_NAME_LABEL
)
from indexing.utils import get_transformations, PIPELINE_CHUNK_SIZE

from llama_index.core.indices.keyword_table.utils import (
    extract_keywords_given_response,
)

from llama_index.graph_stores.neo4j import Neo4jGraphStore
from neo4j import GraphDatabase


from constants import (
    KG_INDEX_LABEL,
    VECTOR_INDEX_LABEL,
)

from llama_index import *
from llama_index.core import StorageContext

neo4j_username = "neo4j"
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_url = "neo4j://localhost:7687"
neo4j_database = "neo4j"


def get_url(host_ip):
    if host_ip != 'localhost':
        return neo4j_url.replace('locahost', host_ip)
    return host_ip


logger = logging.getLogger(__name__)


def create_vector_index(
        nodes, 
        storage_context, 
        num_threads=4,
        pipeline_chunk_size=PIPELINE_CHUNK_SIZE,
        save_path:str ='tmp',
        show_progress=False,
        **kwargs
    ):

    indexed_nodes = run_pipeline_multithreaded(
        nodes, 
        transformations=get_transformations(
            embed=True,
            **kwargs
        ),
        num_threads=num_threads,
        show_progress=show_progress,
        pipeline_chunk_size=pipeline_chunk_size,
    )

    vector_index = VectorStoreIndex(
        nodes=indexed_nodes, 
        storage_context=storage_context,
        show_progress=show_progress
    )

    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/{VECTOR_INDEX_LABEL}.pkl', 'wb') as f:
        pickle.dump(vector_index, f)

    return indexed_nodes, vector_index


def create_keyword_table_index(
        docs, 
        storage_context,
        keyword_extract_template=CODE_KEYWORD_EXTRACT_TEMPLATE_TMPL,
        num_keywords=20,
        num_threads=4,
        show_progress=False,
        save_path:str ='tmp',
        **kwargs
    ) -> VectorStoreIndex:

    transformations = get_transformations(
        keyword_extractor=True,
        keyword_extraction_template=keyword_extract_template,
        num_keywords=num_keywords,
        **kwargs
    )

    indexed_nodes = run_pipeline_multithreaded(
        docs, 
        transformations=transformations,
        num_threads=num_threads,
        show_progress=show_progress
    )

    vector_index = VectorStoreIndex(
        nodes=indexed_nodes, 
        storage_context=storage_context,
        show_progress=show_progress
    )

    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/{KG_INDEX_LABEL}.pkl', 'wb') as f:
        pickle.dump(vector_index, f)

    return indexed_nodes, vector_index
    

def create_knowledge_graph_index(
        docs, 
        graph_nodes, 
        storage_context,
        include_embeddings=False
    ):
    call_graph_links = extract_call_graph_links(graph_nodes)
    index = KnowledgeGraphIndex(
        nodes=[],
        storage_context=storage_context,
        show_progress=True,
    )
    for doc in tqdm(docs, desc="Upserting nodes"):
        class_name = doc.metadata[CLASS_NAME_LABEL]
        graph_links = call_graph_links[class_name]
        triples = set()
        for graph_link in graph_links:
            src = graph_link['src_signature']
            dest = graph_link['dest_signature']
            for triple in [
                (src[CLASS_NAME_LABEL], 'method', src[METHOD_NAME_LABEL]),
                (src[METHOD_NAME_LABEL], 'link', dest[CLASS_NAME_LABEL]),
                (dest[CLASS_NAME_LABEL], 'method', dest[METHOD_NAME_LABEL])
            ]:
                triples.add(triple)
        
        for triple in triples:
            try:
                index.upsert_triplet_and_node(
                    triplet=triple,
                    node=doc,
                    include_embeddings=include_embeddings
                )
            except Exception as e:
                print(f"Error in upserting triplet and node: {triple}")
    
    return index


def get_graph_store(
        db_name,
        host_ip='localhost',
        embed_dim=512,
        hybrid_search=True,
    ):
    create_neo4j_database(db_name)
    url = get_url(host_ip)
    graph_store = Neo4jGraphStore(
        url=neo4j_url,
        username=neo4j_username,
        password=neo4j_password,
        database=neo4j_database,
        embedding_dimension=embed_dim,
        hybrid_search=hybrid_search,
    )
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    return storage_context


def create_neo4j_database(db_name):
    """
    Check if database already exists. If not, create a new database
    """
    uri = neo4j_url
    driver = GraphDatabase.driver(uri, auth=(neo4j_username, neo4j_password))
    with driver.session() as session:
        result = session.run("SHOW DATABASES")
        databases = [record['name'] for record in result]
        if db_name.lower() not in databases:
            # session.run(f"CREATE DATABASE {db_name}")
            print(f"Using the default database for: {db_name}")
    driver.close()
