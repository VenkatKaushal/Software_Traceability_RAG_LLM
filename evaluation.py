import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import pickle
from llama_index.core import Document
from collections import defaultdict
import json
from querying import query_parallel, query_sequential
from retrievers import retrieve_parallel, GraphRetriever
from prompts.templates import CLASS_TRACE_TEMPLATE

from tqdm.auto import tqdm
import json
import re
from typing import Dict, List, Tuple
import uuid
from llama_index.core.schema import TextNode
from llama_index.core.schema import MetadataMode
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.evaluation import (
    BatchEvalRunner,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)


from indexing.utils import run_pipeline_multithreaded, get_transformations
from prompts.templates import REQ2CODE_QA_TEMPLATE
import requests

QUESTIONS_CHUNKS_SIZE = 50

class QADataset(BaseModel):
    """Embedding QA Finetuning Dataset.

    Args:
        queries (Dict[str, str]): Dict id -> query.
        corpus (Dict[str, str]): Dict id -> string.
        relevant_docs (Dict[str, List[str]]): Dict query id -> list of doc ids.

    """

    queries: Dict[str, str] 
    corpus: Dict[str, str]  #
    relevant_docs: Dict[str, List[str]] 
    mode: str = "text"

    @property
    def query_docid_pairs(self) -> List[Tuple[str, List[str]]]:
        """Get query, relevant doc ids."""
        return [
            (query, self.relevant_docs[query_id])
            for query_id, query in self.queries.items()
        ]

    def save_json(self, path: str) -> None:
        """Save json."""
        with open(path, "w") as f:
            json.dump(self.dict(), f, indent=4)

    @classmethod
    def from_json(cls, path: str) -> "QADataset":
        """Load json."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def generate_qa_dataset(
    nodes: List[TextNode],
    qa_generate_prompt_tmpl: str = REQ2CODE_QA_TEMPLATE,
    num_questions_per_chunk: int = 3,
    pipeline_chunk_size=10,
):
    
    if 'questions_this_excerpt_can_answer' not in nodes[0].metadata.keys() == set():
        nodes = run_pipeline_multithreaded(
            nodes,
            transformations=get_transformations(
                num_questions=num_questions_per_chunk,
                questions_template=qa_generate_prompt_tmpl,
            ),
            pipeline_chunk_size=pipeline_chunk_size
        )

    queries = {}
    relevant_docs = {}
    for node in tqdm(nodes):
        node_id = node.node_id
        questions_response = node.metadata['questions_this_excerpt_can_answer']
        result = questions_response.strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0]

        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]

    node_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }

    return QADataset(
        queries=queries, corpus=node_dict, relevant_docs=relevant_docs
    )



# def get_post_processing_results(req_results: Dict):
#     ok = True
#     llm_results = defaultdict(set)
#     for file_name, result in req_results.items():
#         try:
#             class_names_list = json.loads(result.response)
#             for class_name in class_names_list:
#                 llm_results[file_name].add(class_name)
#         except Exception as e:
#             print(file_name, result.response)
#             ok = False

#     for k, value in llm_results.items():
#         llm_results[k] = list(value)
    
#     if not ok:
#         print("Some results are invalid")
#     else:
#         print("All results are valid")

#     return llm_results


# def get_post_processing_results(req_results: Dict) -> Dict[str, List[str]]:
#     GEMINI_API_KEY = "AIzaSyBJ7OItfQQiFbCHb5_EjuOgag5a78p-TSw"
#     GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    
#     req_results_str = str(req_results)

#     input_text = f"""
#     Extract the EA file names and their corresponding class names from the following text.
#     Do not write code. I want a valid JSON as the output. I should be JSON readable output. Do not write in between```json ```.Just JSON output only.
#     Return the output as a valid JSON dictionary, where:
#     - The keys are EA file names (e.g., "EA1.txt", "EA2.txt").
#     - The values are lists of class names extracted after "<<ANSWER>>".

#     Input Text:
#     ---
#     ${req_results_str}
#     ---
#     """

#     print(input_text)
#     headers = {"Content-Type": "application/json"}
#     data = {
#         "contents": [{"parts": [{"text": input_text}]}]
#     }
#     response = requests.post(GEMINI_API_URL, headers=headers, json=data)
#     if response.status_code == 200:
#         result = response.json()
#         generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
#         print("Extracted JSON Output:\n", generated_text)

#         try:
#             parsed_json = json.loads(generated_text)  # Parse JSON properly
#         except json.JSONDecodeError as e:
#             print(f"JSON Decode Error: {e}")
#             parsed_json = {}

#         # Convert to defaultdict(set)
#         llm_results = defaultdict(set)
#         for key, values in parsed_json.items():
#             llm_results[key] = set(values) 

#     else:
#         print(f"Error: {response.status_code}, {response.text}")
#         llm_results = defaultdict(set)
    
#     for k, value in llm_results.items():
#         llm_results[k] = list(value)


#     return llm_results


# def get_post_processing_results(req_results: Dict) -> Dict[str, List[str]]:
#     req_results = f"""
#     {req_results} 
#     """

#     print(req_results)
#     filenames = re.findall(r"'?(EA\d+\.txt)'?", req_results)
#     print(filenames)
#     file_sections = re.split(r"'?(EA\d+\.txt)'?", req_results)[1:]
#     filename_class_map = {}
    
#     for i in range(0, len(file_sections), 2):
#         filename = file_sections[i].strip()
#         section = file_sections[i + 1] if i + 1 < len(file_sections) else ""
        
#         # Extract class names between the filename and next filename or end of text
#         classnames = re.findall(r"<<ANSWER>>\s*(\[.*?\])", section, re.DOTALL)
#         print(classnames)
#         filename_class_map[filename] = classnames[0] if classnames else []
    
#     parsed_json = filename_class_map

#     # Convert to defaultdict(set)
#     llm_results = defaultdict(set)
#     for key, values in parsed_json.items():
#         llm_results[key] = set(values) 

    
#     for k, value in llm_results.items():
#         llm_results[k] = list(value)


#     return llm_results

def get_post_processing_results(req_results: Dict) -> Dict[str, List[str]]:
    req_results = f"""
    {req_results} 
    """

    print("Raw Input:\n", req_results)
    
    filenames = re.findall(r"'?(EA\d+\.txt)'?", req_results)
    print("Extracted Filenames:", filenames)
    
    file_sections = re.split(r"'?(EA\d+\.txt)'?", req_results)[1:]
    filename_class_map = {}
    
    for i in range(0, len(file_sections), 2):
        filename = file_sections[i].strip()
        section = file_sections[i + 1] if i + 1 < len(file_sections) else ""
        
       
        classnames_match = re.findall(r"<<ANSWER>>.*?\[([^\]]+)\]", section, re.DOTALL)
        
        if classnames_match:
            try:
                classnames = [name.strip().strip('"') for name in classnames_match[0].split(',')]
            except json.JSONDecodeError:
                classnames = []
        else:
            classnames = []
        
        print(f"Extracted classes for {filename}:", classnames)
        filename_class_map[filename] = list(set(classnames))

    return filename_class_map


def get_solutions(file_name):
    gts = [line for line in open(file_name).read().split('\n') if line]
    solutions = defaultdict(list)
    for gt in gts:
        gt_split = gt.split(': ')
        file_name = gt_split[0]
        ext = '.java' if gt_split[1].endswith('.java') else '.txt'
        class_name = gt_split[1].split(ext)[0]
        solutions[file_name].append(class_name)
    return solutions


def compare_solutions(
        solutions, 
        llm_results, 
    ):
    results = list()
    tp, fp = 0, 0
    tn, fn = 0, 0
    for file_name, classes in solutions.items():
        if file_name in llm_results:
            tp += len(set(classes).intersection(set(llm_results[file_name])))
            fp += len(set(llm_results[file_name]) - set(classes))
            fn += len(set(classes) - set(llm_results[file_name]))
            
            result = {
                "file_name": file_name,
                "expected_classes": sorted(classes),
                "llm_classes": sorted(llm_results[file_name])
            }
            results.append(result)
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    scores = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    return results, scores


def evaluate_from_retriever(
        req_nodes: List[Document],
        retrievers: Dict,
        solutions_file: str,
        dataset_name: str = 'tmp',
        results_dir: str = 'results',
    ):
    results = dict()
    solutions = get_solutions(solutions_file)
    for config, retriever in retrievers.items():
        print(f"Evaluating for {config}")
        req_results = retrieve_parallel(
            retriever, 
            req_nodes, 
            CLASS_TRACE_TEMPLATE,
            num_threads=8,
        )
        print(f"Results for {config}")
        config_results = compare_solutions(solutions, req_results)
        results[config] = config_results

        with open(f'{results_dir}/{dataset_name}_{config}_retriever_correctness_results.json', 'w') as f:
            json.dump(config_results, f, indent=4)


count = 0
def evaluate_response(
        req_nodes: List[Document], 
        query_engines: dict,
        solutions_file: str,
        dataset_name: str = 'tmp',
        results_dir: str = 'results',
    ):
    global count
    results = dict()
    print("Getting solutions")
    solutions = get_solutions(solutions_file)
    print("Solutions obtained")
    for config, query_engine in query_engines.items():
        print(f"Evaluating for {config}")
        req_results = query_parallel( 
            query_engine, 
            CLASS_TRACE_TEMPLATE, 
            req_nodes, 
            num_threads=8,
        )
        print("Evaluating parallel ended")
        llm_results = req_results
        print(req_results)
        if not isinstance(query_engine, GraphRetriever):
            llm_results = get_post_processing_results(req_results)
        traceability_file = f"{results_dir}/{dataset_name}_{config}_traceability_link_{count}.json"
        with open(traceability_file, 'w') as trace_file:
            json.dump(llm_results, trace_file, indent=4)
        print(f"Traceability links saved in {traceability_file}")
        count = count + 1
        print(f"Results for {config}")
        config_results = compare_solutions(solutions, llm_results)
        print("Results compared")
        results[config] = config_results
        print("Storing Results")
        with open(f'{results_dir}/{dataset_name}_{config}_correctness_results.json', 'w') as f:
            json.dump(config_results, f, indent=4)

    return results



def get_questions_from_nodes(nodes):
    print("RAG Dataset Generation")
    
    context_str = "\n".join([node.text for node in nodes])  
    formatted_query = REQ2CODE_QA_TEMPLATE.format(
        context_str=context_str,
        num_questions_per_chunk=1
    )
    dataset_generator = RagDatasetGenerator.from_documents(
        nodes,
        num_questions_per_chunk=1,
        show_progress=True,
        question_gen_query=formatted_query,
        workers=8,
    )
    print("RAG  Dataset Generation Completed")
    rag_dataset = dataset_generator.generate_questions_from_nodes()
    print("Questions Generated")
    questions = [e.query for e in rag_dataset.examples]
    return questions


def evaluate_index(
        questions: List[str],
        query_engine,
        num_workers: int = 4,
):
    runner = BatchEvalRunner(
        {
            "faithfulness": FaithfulnessEvaluator(),
            "relevancy": RelevancyEvaluator(),
        },
        workers=num_workers,
    )

    eval_results = runner.evaluate_queries(
        query_engine, queries=questions
    )
    return eval_results


def evaluate_questions_on_index(
        questions: List, 
        qe,
        num_workers: int = 4,
        num_threads: int = 4,
    ):
    question_chunks = [
        questions[i:i + QUESTIONS_CHUNKS_SIZE] \
            for i in range(0, len(questions), QUESTIONS_CHUNKS_SIZE)
    ]

    results = list()
    total_jobs = len(question_chunks)
    
    with tqdm(total=total_jobs, desc=f"Evaluating Question Chunks of {QUESTIONS_CHUNKS_SIZE}") as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(evaluate_index, chunk, qe, num_workers) for chunk in question_chunks]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
    
    eval_results = dict()
    eval_results['faithfulness'] = [r for result in results for r in result['faithfulness']]
    eval_results['relevancy'] = [r for result in results for r in result['relevancy']]

    return eval_results


def evaluate_retrieval(
        questions: List[str], 
        query_engines,
        dataset_name: str = 'tmp',
        num_workers: int = 8,
    ):
    extract_result = lambda results, key: sum(result.passing for result in results[key]) / len(results[key])
    results = dict()
    for index_name, qe in query_engines.items():
        print(f"Evaluating {index_name}")
        eval_results = evaluate_questions_on_index(questions, qe, num_workers)
        faithfulness = extract_result(eval_results, "faithfulness")
        relevancy = extract_result(eval_results, "relevancy")
        results[index_name] = {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
        }

        with open(f'results/{dataset_name}_{index_name}_evaluation_results.pkl', 'wb') as f:
            pickle.dump(eval_results, f)

        with open(f'results/{dataset_name}_{index_name}_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)

    return results