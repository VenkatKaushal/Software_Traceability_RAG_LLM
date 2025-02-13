from concurrent.futures import ThreadPoolExecutor
from llama_index.llms.anyscale import Anyscale
from llama_index.core.llms import ChatMessage

from llama_index.core.schema import NodeRelationship
from llama_index.core import Document
from tqdm.auto import tqdm

from prompts.templates import (
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_TEMPLATE
)

SUMMARY_SYSTEM_PROMPT = \
"""
You are an expert software engineer who has been asked to generate a summary of a class in a Java code file. The class contains attributes and methods.
"""

SUMMARY_TEMPLATE = \
"""
Given a class in a Java code file, generate a class summary to map a given use case requirements to the given java code. 
The summary should capture the purpose of this class such that given a use case requirement, it can be determined if this class is relevant.
The summary should be concise not more than 2-3 lines which contains Java Code Keywords present in the class that can be useful to map a usecase requirement to this java code.
"""

def get_llm_chat_response(
        prompt: str, 
        llm: Anyscale,
        system_prompt: str = SUMMARY_SYSTEM_PROMPT,
    ):
    system_message = ChatMessage(role="system", content=f"{system_prompt}")
    prompt_message = ChatMessage(role="user", content=f"{SUMMARY_TEMPLATE}\n\n{prompt}")

    response = llm.chat([system_message, prompt_message])
    return response.message.content


def get_document_text(
        node: Document, 
        nodes_map: dict[str, Document]
    ):
    text = ""
    visited_nodes = set()
    while node:
        visited_nodes.add(node.id_)
        text += node.text + "\n"
        next_node = node.relationships.get(NodeRelationship.NEXT, None)
        node = nodes_map.get(next_node.node_id, None) if next_node else None

    
    return text, visited_nodes


def add_summary_to_node_groups(
        node_groups, 
        llm,
        num_workers: int = 4
    ):
    summaries = list()
    progress_bar = tqdm(total=len(node_groups), desc="Adding Summaries")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = list()
        for _, (node_group) in enumerate(node_groups):
            futures.append(
                executor.submit(add_summary_to_node_group, node_group, llm)
            )
        
        for future in futures:
            summaries += future.result()
            progress_bar.update(1)

    return summaries


def add_summary_to_node_group(node_chunk, llm):
    doc_text, nodes = node_chunk["text"], node_chunk["nodes"]
    summary = get_llm_chat_response(doc_text, llm)
    for node in nodes:
        node.metadata["summary"] = summary
    
    return summary


def add_node_summaries(
        nodes: list[Document], 
        llm
    ):
    nodes_map = {node.id_: node for node in nodes}
    summary_chunks = list()
    all_visited_nodes = set()
    for node in nodes:
        if node.id_ in all_visited_nodes:
            continue
        doc_text, visited_nodes = get_document_text(node, nodes_map)
        summary_chunks.append(
            {
                "text": doc_text,
                "nodes": [nodes_map[node] for node in visited_nodes]
            }
        )
        
        all_visited_nodes.update(visited_nodes)
    
    print("Number of Summary Chunks: ", len(summary_chunks))
    summaries = add_summary_to_node_groups(summary_chunks, llm)
    return summaries