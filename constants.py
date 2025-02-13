from enum import Enum


class LLMs(Enum):
    MISTRAL_8X7B = 'mistral8x7b'
    MISTRAL_8X22B = 'mistral8x22b'
    MISTRAL_7B = 'mistral7b'
    CODELLAMA = "codellama"
    LLAMA3 = "llama3"
    GOOGLE_GEMMA = "google_gemma"
    COMMANDR_PLUS = 'command-r-plus'
    COMMANDR = 'command-r'
    COMMAND = 'command'
    GPT_4 = "gpt-4"
    GPT_3 = "gpt-3"
    DEFAULT_LLM = LLAMA3


LLMsMap = {
    LLMs.MISTRAL_8X7B.value: "mistralai/Mixtral-8x7B-Instruct-v0.1",
    LLMs.MISTRAL_8X22B.value: "mistralai/Mixtral-8x22B-Instruct-v0.1",
    LLMs.MISTRAL_7B.value: "mistralai/Mixtral-7B-Instruct-v0.1",
    LLMs.CODELLAMA.value: "codellama/CodeLlama-70b-Instruct-hf",
    LLMs.LLAMA3.value: "google/gemma-2-2b",
    LLMs.GOOGLE_GEMMA.value: "google/gemma-7b-it",
    LLMs.COMMANDR_PLUS.value: "command-r-plus",
    LLMs.COMMANDR.value: "command-r",
    LLMs.COMMAND.value: "command",
    LLMs.GPT_4.value: "gpt-4-1106-preview",
    LLMs.GPT_3.value: "gpt-3.5-turbo"
}


class EmbeddingModels(Enum):
    HF_BGE_LARGE = "bge_large"
    HF_BGE_BASE = "bge_base"
    HF_BGE_SMALL = "bge_small"
    HF_BGE_M3 = "bge_m3"
    AS_THENLPER = "gte_large"
    DEFAULT_EMBED_MODEL = HF_BGE_SMALL


EmbeddingModelsMap = {
    EmbeddingModels.HF_BGE_LARGE.value: "BAAI/bge-large-en-v1.5",
    EmbeddingModels.HF_BGE_BASE.value: "BAAI/bge-base-en-v1.5",
    EmbeddingModels.HF_BGE_SMALL.value: "BAAI/bge-small-en-v1.5",
    EmbeddingModels.HF_BGE_M3.value: "BAAI/bge-m3",
    EmbeddingModels.AS_THENLPER.value: "thenlper/gte-large",
}


class LLMTypes(Enum):
    ANY_SCALE = "anyscale"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    OPENAI = "openai"
    DEFAULT_LLM = ANY_SCALE

KG_INDEX_LABEL = 'kg_index'
KG_INDEX_CUSTOM_LABEL = 'kg_index_custom'
VECTOR_INDEX_LABEL = 'vector_index'
KW_TABLE_LABEL = 'kw_table'
CODE_KG_INDEX_LABEL = 'code_kg_index.pkl'
SIMPLE_KW_TABLE_LABEL = 'simple_kw_table.pkl'


VECTOR_INDEX_RETREIVER = 'vir'
BM25_INDEX_RETREIVER = 'bm25'