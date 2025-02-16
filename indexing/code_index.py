import ast
from collections import defaultdict
from typing import Dict, List
from llama_index.core.schema import (
    Document, 
    NodeRelationship, 
    RelatedNodeInfo
)
from tqdm.auto import tqdm
from indexing.constants import (
    FILE_NAME_LABEL,
    CLASS_NAME_LABEL,
    DOCSTRING_LABEL,
    ATTRIBUTES_LABEL,
    ATTRIBUTE_NAME_LABEL,
    ATTRIBUTES_TYPE_LABEL,
    METHOD_NAME_LABEL,
    METHOD_SIGNATURE_LABEL,
    METHOD_DOCSTRING,
    CALLS_LABEL,
    CALLED_BY_LABEL,
    METHODS_LABEL,
)

from indexing.constants import (
    CODE_METHODS,
    CODE_METHOD_CALLS
)


def create_class_node_doc(
        graph_node, 
        add_method_calls=True
    ):

    metadata = {
        FILE_NAME_LABEL: graph_node[FILE_NAME_LABEL],
        CLASS_NAME_LABEL: graph_node[CLASS_NAME_LABEL],
    }

    content = f"{graph_node['type']} Name: " + graph_node[CLASS_NAME_LABEL] + "\n"
    
    if ATTRIBUTES_LABEL in graph_node and len(graph_node[ATTRIBUTES_LABEL]):
        content += f"Attributes: \n"
        for attr in graph_node[ATTRIBUTES_LABEL]:
            content += f"{attr[ATTRIBUTE_NAME_LABEL]}: {attr[ATTRIBUTES_TYPE_LABEL]}\n"
    
    if DOCSTRING_LABEL in graph_node:
        content += f"\n{graph_node[DOCSTRING_LABEL]}\n"
    
    content += "\nMethods: \n"
    method_calls = dict()
    method_nodes = graph_node[METHODS_LABEL]
    for method_node in method_nodes:
        content += f"\nMethod Name: {method_node[METHOD_NAME_LABEL]}\n"
        content += f"Signature: {method_node[METHOD_SIGNATURE_LABEL]}\n"
        content += f"Class Name: {graph_node[CLASS_NAME_LABEL]}\n"
        if METHOD_DOCSTRING in method_node:
            content += f"Docstring: \n{method_node[METHOD_DOCSTRING]}\n"\
            if method_node[METHOD_DOCSTRING] else f""

        if add_method_calls:
            calls = list()
            if CALLS_LABEL in method_node and len(method_node[CALLS_LABEL]):
                
                content += f"\nCalls: \n"
                for call in method_node[CALLS_LABEL]:
                    content += f"{method_node[METHOD_SIGNATURE_LABEL]} calls {call}\n"
                    calls.append(call)
            
            called_bys = list()
            if CALLED_BY_LABEL in method_node and len(method_node[CALLED_BY_LABEL]):
                content += f"\nCalled By: \n"
                for called_by in method_node[CALLED_BY_LABEL]:
                    content += f"{method_node[METHOD_SIGNATURE_LABEL]} called by {called_by}\n"
                    called_bys.append(called_by)

            method_calls[method_node[METHOD_NAME_LABEL]] = {
                CALLS_LABEL: calls,
                CALLED_BY_LABEL: called_bys
            }

        content += "\n\n"

    metadata[CODE_METHODS] = str(graph_node[METHODS_LABEL])
    metadata[CODE_METHOD_CALLS] = str(method_calls)

    doc = Document(
        text=content,
        metadata=metadata,
        excluded_embed_metadata_keys=[FILE_NAME_LABEL, CODE_METHOD_CALLS, CODE_METHODS],
        excluded_llm_metadata_keys=[FILE_NAME_LABEL, CODE_METHOD_CALLS, CODE_METHODS]
    )
    return doc


def create_code_graph_nodes_index(
        graph_nodes, 
        add_method_calls=True
    ):
    docs = [
        create_class_node_doc(graph_node, add_method_calls) \
        for graph_node in tqdm(graph_nodes.values(), desc="Creating Documents")
        if graph_node["type"] == "Class"
    ]
    
    print("Number of documents indexed:", len(docs))
    return docs


def add_node_relationships(
        node: Document, 
        relationship_type: NodeRelationship, 
        related_nodes: List[RelatedNodeInfo]
    ):
    if relationship_type not in node.relationships:
            node.relationships[relationship_type] = list()
        
    if not isinstance(node.relationships[relationship_type], list):
        node.relationships[relationship_type] = [node.relationships[relationship_type]]
    
    node.relationships[relationship_type].extend(related_nodes)


def add_method_call_links_to_docs(
        docs: List[Document],
        rel_type = 'next'
    ):

    if rel_type == 'next':
        src_rel_type = NodeRelationship.NEXT
        dest_rel_type = NodeRelationship.PREVIOUS
    elif rel_type == 'parent':
        src_rel_type = NodeRelationship.PARENT
        dest_rel_type = NodeRelationship.CHILD


    new_links_count = 0
    node_map = {n.hash: n for n in docs}
    mc2node_map: Dict[str, Document] = dict()
    for node in docs:
        methods = ast.literal_eval(node.metadata[CODE_METHODS])
        for method in methods:
            mc2node_map[method[METHOD_SIGNATURE_LABEL]] = node

    call_links, called_by_links = defaultdict(set), defaultdict(set)
    for node in tqdm(docs, desc="Adding Method Call Links to Docs"):
        methods_calls = ast.literal_eval(node.metadata[CODE_METHOD_CALLS])
        methods = ast.literal_eval(node.metadata[CODE_METHODS])
        for method in methods:
            if method[METHOD_NAME_LABEL] not in methods_calls:
                continue
            method_calls = methods_calls[method[METHOD_NAME_LABEL]]
            calls, called_bys = method_calls[CALLS_LABEL], method_calls[CALLED_BY_LABEL]
            for call in calls:
                if call in mc2node_map and mc2node_map[call].hash != node.hash:
                    call_links[node.hash].add(mc2node_map[call].hash)
            
            for called_by in called_bys:
                if called_by in mc2node_map and mc2node_map[called_by].hash != node.hash:
                    called_by_links[node.hash].add(mc2node_map[called_by].hash)
    
    
    for node_hash, related_nodes in call_links.items():
        node = node_map[node_hash]
        related_called_nodes = [node_map[related_node].as_related_node_info() for related_node in related_nodes]
        add_node_relationships(node, src_rel_type, related_called_nodes)
        new_links_count += len(related_nodes)

    for node_hash, related_nodes in called_by_links.items():
        node = node_map[node_hash]
        related_called_by_nodes = [node_map[related_node].as_related_node_info() for related_node in related_nodes]
        add_node_relationships(node, dest_rel_type, related_called_by_nodes)
        new_links_count += len(related_nodes)
    
    print("Number of new links added:", new_links_count)
