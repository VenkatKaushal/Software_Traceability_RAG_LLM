"""
This script processes nodes through a series of transformations, such as summarization, keyword extraction, and question generation, and supports multithreaded execution for efficient handling of large datasets. It also integrates with external graph stores like Kuzu and Chroma for data storage.

Functions and Classes:

1. change_api_key:
   - Changes the API key for a specified index. This is used to set the appropriate API key when switching between different LLMs or embedding models.

2. get_transformations:
   - Returns a list of transformations to apply to the nodes, such as embedding, summary extraction, title extraction, question answering, and keyword extraction. Each transformation is based on the provided parameters like LLM, number of questions, etc.

3. get_pipeline_chunks:
   - Breaks the nodes into smaller chunks to process in parallel. Each chunk is processed by an ingestion pipeline, and transformations are applied to them.

4. run_pipeline:
   - Runs a given ingestion pipeline on the provided nodes, applying the specified transformations and returning the processed nodes.

5. run_pipeline_multithreaded:
   - Processes nodes in parallel using a thread pool. It splits the work into chunks and uses multiple threads to process them, improving performance for large datasets.

6. get_parser:
   - Returns a `LangchainNodeParser` object configured with a text splitter for breaking documents into smaller chunks, based on the specified chunk size and overlap.

7. get_vector_storage_context:
   - Creates a Chroma database for vector storage. This function connects to a persistent database and returns a storage context.

8. get_kuzu_graph_store:
   - Creates a Kuzu graph store from the provided collection name, and returns the storage context for graph data.

9. summarize_nodes:
   - Summarizes nodes using a provided summary template. It processes nodes through a summarization pipeline and applies transformations to extract summaries from the content.

10. generate_queries:
   - Generates queries based on a given node using a prompt template. The queries are then added to the node's metadata.

11. create_semantically_similar_nodes:
   - Creates semantically similar nodes by generating queries for a list of nodes. The function processes nodes in parallel, generating queries for each node.

12. embed_nodes:
   - Embeds the provided nodes by first adding summaries to the nodes and then running a pipeline with an embedding transformation. The processed nodes are saved to a file.

Dependencies:
- pickle: For saving processed nodes.
- kuzu, chromadb: For graph storage solutions.
- llama_index: Provides classes and functions for working with nodes, documents, and embeddings.
- langchain: Used for processing documents and splitting them into smaller chunks.
- transformers: For generating queries and working with LLMs.
- tqdm.auto: For displaying progress bars during processing.
"""


import pickle
import kuzu
from llama_index.graph_stores.kuzu import KuzuGraphStore
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import List, Union
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Settings
from llama_index.core import Document
from llama_index.core.schema import TextNode, MetadataMode

from tqdm.auto import tqdm

from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    KeywordExtractor
)

from llama_index.core.node_parser import LangchainNodeParser
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter
)

import chromadb
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore



from llama_index.core.extractors.metadata_extractors import (
    DEFAULT_QUESTION_GEN_TMPL,
    DEFAULT_TITLE_NODE_TEMPLATE
)

from prompts.templates import CODE_SUMMARY_EXTRACT_TEMPLATE, SCENARIO_GEN_TEMPLATE

from api_models import set_llm
from prompts.templates import CODE_KEYWORD_EXTRACT_TEMPLATE_TMPL

CHUNK_SIZE = 128
CHUNK_OVERLAP = 25

PIPELINE_CHUNK_SIZE = 10


def change_api_key(idx):
    print("Changing API Key")
    api_key = get_api_keys(llm_type='anyscale', idx=idx)
    c = Settings.llm_embed_config
    llm_type, llm_name = c['llm_type'], c['llm_name']
    set_llm(model_type=llm_type, model_name=llm_name, api_key=api_key)


def get_transformations(
    llm=None,
    embed=False,
    embed_model=None,
    summary_extractor=False,
    summary_template=CODE_SUMMARY_EXTRACT_TEMPLATE,
    num_questions=None, 
    questions_template=DEFAULT_QUESTION_GEN_TMPL,
    num_title_nodes=None,
    title_template=DEFAULT_TITLE_NODE_TEMPLATE,
    qa_prompt=None,
    keyword_extractor=False,
    keyword_extraction_template=CODE_KEYWORD_EXTRACT_TEMPLATE_TMPL,
    num_keywords=None
):
    llm = llm if llm is not None else Settings.llm
    transformations = list()

    if embed:
        transformations.append(Settings.embed_model if embed_model is None else embed_model)

    if summary_extractor:
        
        summary_extractor = SummaryExtractor(
            llm=llm,
            prompt_template=summary_template
        )
            
        transformations.append(summary_extractor)
    
    if num_title_nodes is not None:
        title_extractor = TitleExtractor(
            llm=llm,
            nodes=num_title_nodes,
            prompt_template=title_template
        )
        transformations.append(title_extractor)

    if num_questions is not None:
        qa_extractor = QuestionsAnsweredExtractor(
            llm=llm,
            questions=num_questions,
            prompt_template=questions_template
        )
        if qa_prompt is not None:
            qa_extractor.prompt_template = qa_prompt
        
        transformations.append(qa_extractor)
    
    if keyword_extractor:
        prompt_template=CODE_KEYWORD_EXTRACT_TEMPLATE_TMPL \
            if keyword_extraction_template else keyword_extraction_template
        kw_extractor = KeywordExtractor(
            llm=llm,
            prompt_template=prompt_template,
            keywords=num_keywords
        )
        transformations.append(kw_extractor)
    
    assert len(transformations) > 0, "No transformations provided"

    return transformations


def get_pipeline_chunks(
        nodes: Union[Document, TextNode],
        transformations: list=None,
        parser: LangchainNodeParser = None,
        pipeline_chunk_size:int = PIPELINE_CHUNK_SIZE
    ):

    transformations = transformations if transformations is not None else list()
    pipeline_chunks = list()

    for i in tqdm(range(0, len(nodes), pipeline_chunk_size), desc='Creating nodes'):
        docs = nodes[i:i+pipeline_chunk_size]
        if parser:
            docs = parser.get_nodes_from_documents(docs, show_progress=True)
        
        pipeline = IngestionPipeline(transformations=transformations)
        pipeline_chunks.append((pipeline, docs))
    
    return pipeline_chunks


def run_pipeline(pipeline: IngestionPipeline, docs, show_progress=True):

    try:
        index_nodes = pipeline.run(
            nodes=docs, 
            show_progress=show_progress
        )
    except Exception as e:
        print(e)
        index_nodes = []
        
    return index_nodes


def run_pipeline_multithreaded(
        nodes, 
        transformations: list = list(),
        num_threads=2,
        pipeline_chunk_size=PIPELINE_CHUNK_SIZE,
        show_progress=True
    ) -> List[TextNode]:

    pipeline_chunks = get_pipeline_chunks(
        nodes,
        transformations=transformations,
        pipeline_chunk_size=pipeline_chunk_size
    )

    # pipeline, docs = pipeline_chunks[0]
    # run_pipeline(pipeline, docs, show_progress)

    total_jobs = len(pipeline_chunks)
    indexed_nodes = list()
    with tqdm(total=total_jobs, desc=f"Executing Chunks of {pipeline_chunk_size}") as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = list()
            for _, (pipeline_docs) in enumerate(pipeline_chunks):
                pipeline, docs = pipeline_docs
                futures.append(
                    executor.submit(run_pipeline, pipeline, docs, show_progress)
                )
            
            for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    indexed_nodes += result
                    pbar.update(1)
        print("In Multithreaded pipeline")

    return indexed_nodes


def get_parser(
        language: Language = None,
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
):

    if language:
        splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            language=language
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    parser = LangchainNodeParser(splitter)
    return parser


def get_vector_storage_context(chroma_db_path, collection_name):
    db = chromadb.PersistentClient(path=f"{chroma_db_path}")
    chroma_collection = db.get_or_create_collection(collection_name)
    print(f"Chroma Collection: {chroma_collection}")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context, vector_store


def get_kuzu_graph_store(collection_name):
    db = kuzu.Database(collection_name)
    graph_store = KuzuGraphStore(db)

    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    return storage_context


def summarize_nodes(
        nodes,
        summary_template,
        chunk_size=8192,
        chunk_overlap=100,
        num_threads=8,
        pipeline_chunk_size=PIPELINE_CHUNK_SIZE
):
    summarization_parser = get_parser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = summarization_parser.get_nodes_from_documents(nodes)
    summarized_nodes = run_pipeline_multithreaded(
        nodes=docs,
        transformations=get_transformations(
            summary_extractor=True,
            summary_template=summary_template,
        ),
        pipeline_chunk_size=pipeline_chunk_size,
        num_threads=num_threads,
    )
    return summarized_nodes


def generate_queries(
        node: TextNode, 
        num_queries: int = 5,
        query_gen_prompt: str = SCENARIO_GEN_TEMPLATE
    ):
    llm = Settings.llm
    fmt_prompt = query_gen_prompt.format(
        num_queries=num_queries, query=node.get_content(MetadataMode.LLM)
    )
    response = llm.complete(fmt_prompt)
    useless = f'Here are'
    queries = [r for r in response.text.split("\n") if r.strip() and useless.lower() not in r]
    node.metadata['similar_queries'] = queries
    return node



def create_semantically_similar_nodes(
        nodes: List[TextNode],
        num_queries: int = 5,
        query_gen_prompt: str = SCENARIO_GEN_TEMPLATE,
        num_threads=8,
):
    total_jobs = len(nodes)
    results = list()
    with tqdm(total=total_jobs, desc=f"Creating Similar Nodes") as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(
                generate_queries, node, num_queries, query_gen_prompt) 
                for node in nodes
            ]
            
            for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)
    return results


def embed_nodes(nodes, save_file):
    for i in range(len(nodes)):
        nodes[i].text += f"\nSummary:\n{nodes[i].metadata['section_summary']}"
        nodes[i].excluded_embed_metadata_keys += ['section_summary']
        

    nodes = get_parser().get_nodes_from_documents(nodes)
    indexed_nodes = run_pipeline_multithreaded(
        nodes, 
        transformations=get_transformations(
            embed=True,
        ),
        num_threads=8,
        show_progress=True,
    )
    with open(f'indexed_nodes/{save_file}_nodes.pkl', 'wb') as f:
        pickle.dump(indexed_nodes, f)
    
    print(f"Number of indexed nodes embedded: {len(indexed_nodes)}")