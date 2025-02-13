import pickle
from api_models import get_api_keys, set_llm_and_embed
from constants import EmbeddingModelsMap, LLMsMap
from parameters import parse_args
from run import get_code_nodes
from indexing.utils import get_parser, summarize_nodes
from prompts.templates import code_summarization_template


def summarize(args, dataset_name):
    code_nodes = get_code_nodes(args, dataset_name)
    # parser = get_parser()
    # print(f"Getting nodes from documents...")
    # docs = parser.get_nodes_from_documents(code_nodes)
    # print(f"Number of nodes: {len(docs)}")
    summary_template = code_summarization_template(args.add_method_calls)

    summarized_nodes = summarize_nodes(
        code_nodes,
        summary_template=summary_template,
        pipeline_chunk_size=args.num_threads
    )

    with open(f'summarized_nodes/{dataset_name}{"_mc" if args.add_method_calls else ""}.pkl', 'wb') as f:
        pickle.dump(summarized_nodes, f)
    
    return summarized_nodes


def main():
    configs = [
        ('iTrust', 'bge_large'),
        ('smos', 'bge_m3'),
        ('eANCI', 'bge_m3'),
        ('eTour', 'bge_large')
    ]
    args = parse_args()

    for i, (dataset_name, embed_model) in enumerate(configs):
        api_key = get_api_keys(llm_type=args.llm_type, idx=i)
        args.embed_model = embed_model
        llm_name = LLMsMap[args.llm]
        embed_model_name = EmbeddingModelsMap[embed_model]
        print(f"Using LLM: {llm_name}")
        print(f"Using Embedding Model: {embed_model_name}")

        set_llm_and_embed(
            llm_type=args.llm_type,
            llm_name=llm_name,
            embed_model_name=embed_model_name,
            api_key=api_key
        )

        summarize(args, dataset_name)

if __name__ == '__main__':
    main()