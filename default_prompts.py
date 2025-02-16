"""Set of default prompts refined for LLM to avoid prompt repetition."""    
""" Replace this file with the one in: llama-index-core/llama_index/core/prompts/default_prompts.py """

from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

############################################
# 1) Summarization Prompts
############################################

DEFAULT_SUMMARY_PROMPT_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Read the following text carefully and write a concise summary using only the provided information. "
    "Include key details, but do NOT repeat these instructions or the text verbatim.\n\n"
    "{context_str}\n"
    "<<ANSWER>>\n"
)

DEFAULT_SUMMARY_PROMPT = PromptTemplate(
    DEFAULT_SUMMARY_PROMPT_TMPL, prompt_type=PromptType.SUMMARY
)


############################################
# 2) Insert Prompt
############################################

DEFAULT_INSERT_PROMPT_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Below is a numbered list of context (1 to {num_chunks}). Each item corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "---------------------\n"
    "Here is a new piece of information: {new_chunk_text}\n\n"
    "Identify the single most relevant summary to update. Respond ONLY with the number.\n"
    "Do NOT repeat these instructions.\n"
    "<<ANSWER>>\n"
)

DEFAULT_INSERT_PROMPT = PromptTemplate(
    DEFAULT_INSERT_PROMPT_TMPL, prompt_type=PromptType.TREE_INSERT
)


############################################
# 3) Single-Choice Query Prompt
############################################

DEFAULT_QUERY_PROMPT_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Below is a numbered list of summaries (1 to {num_chunks}).\n"
    "---------------------\n"
    "{context_list}\n"
    "---------------------\n"
    "Based only on the above choices (no prior knowledge), find the ONE most relevant summary to answer:\n"
    "'{query_str}'\n\n"
    "Then respond in the format:\n"
    "1) The choice number (e.g. `ANSWER: 3`)\n"
    "2) A brief explanation why that summary addresses the question.\n"
    "Do NOT repeat these instructions.\n"
    "<<ANSWER>>\n"
)

DEFAULT_QUERY_PROMPT = PromptTemplate(
    DEFAULT_QUERY_PROMPT_TMPL, prompt_type=PromptType.TREE_SELECT
)


############################################
# 4) Multiple-Choice Query Prompt
############################################

DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Below is a numbered list of summaries (1 to {num_chunks}).\n"
    "---------------------\n"
    "{context_list}\n"
    "---------------------\n"
    "Based ONLY on the above choices (no prior knowledge), choose up to {branching_factor} summaries (ranked most relevant first) for:\n"
    "'{query_str}'\n\n"
    "Format:\n"
    "1) `ANSWER: <numbers>`\n"
    "2) Briefly explain why these were selected.\n"
    "Do NOT repeat these instructions.\n"
    "<<ANSWER>>\n"
)

DEFAULT_QUERY_PROMPT_MULTIPLE = PromptTemplate(
    DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL, prompt_type=PromptType.TREE_SELECT_MULTIPLE
)


############################################
# 5) Refine Prompt
############################################

DEFAULT_REFINE_PROMPT_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Original query: {query_str}\n"
    "Existing answer: {existing_answer}\n\n"
    "Additional context:\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Refine the existing answer if needed. If the context is irrelevant, keep the original.\n"
    "Do NOT repeat these instructions.\n"
    "<<ANSWER>>\n"
)

DEFAULT_REFINE_PROMPT = PromptTemplate(
    DEFAULT_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)


############################################
# 6) Text Q&A Prompt
############################################

DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Context information is below:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using ONLY this context (no prior knowledge), answer:\n"
    "{query_str}\n\n"
    "Do NOT repeat these instructions.\n"
    "<<ANSWER>>\n"
)

DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)


############################################
# 7) Tree Summarize Prompt
############################################

DEFAULT_TREE_SUMMARIZE_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Context from multiple sources is below:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Based on these sources (no prior knowledge), answer:\n"
    "{query_str}\n\n"
    "Do NOT repeat these instructions.\n"
    "<<ANSWER>>\n"
)

DEFAULT_TREE_SUMMARIZE_PROMPT = PromptTemplate(
    DEFAULT_TREE_SUMMARIZE_TMPL, prompt_type=PromptType.SUMMARY
)


############################################
# 8) Keyword Extraction
############################################

DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Extract up to {max_keywords} keywords from the text below (avoid stopwords):\n"
    "---------------------\n"
    "{text}\n"
    "---------------------\n"
    "Return them in comma-separated form (e.g., `KEYWORDS: keyword1, keyword2`).\n"
    "Do NOT repeat these instructions.\n"
    "<<ANSWER>>\n"
)

DEFAULT_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL, prompt_type=PromptType.KEYWORD_EXTRACT
)


############################################
# 9) Query Keyword Extraction
############################################

DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "A question is provided below. Extract up to {max_keywords} keywords from the question to help with lookup.\n"
    "Avoid stopwords.\n"
    "---------------------\n"
    "{question}\n"
    "---------------------\n"
    "Return them in comma-separated form (e.g., `KEYWORDS: keyword1, keyword2`).\n"
    "Do NOT repeat these instructions.\n"
    "<<ANSWER>>\n"
)

DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)


############################################
# 10) Schema Extraction
############################################

DEFAULT_SCHEMA_EXTRACT_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "We have an unstructured text and a structured schema.\n"
    "-----------TEXT-----------\n"
    "{text}\n"
    "-----------SCHEMA-----------\n"
    "{schema}\n"
    "---------------------\n"
    "Extract fields in the form:\n"
    "field1: <value>\n"
    "field2: <value>\n"
    "...\n\n"
    "If a field isn’t present, skip it. If none are present, return a blank string.\n"
    "Do NOT repeat these instructions.\n"
    "<<ANSWER>>\n"
)

DEFAULT_SCHEMA_EXTRACT_PROMPT = PromptTemplate(
    DEFAULT_SCHEMA_EXTRACT_TMPL, prompt_type=PromptType.SCHEMA_EXTRACT
)


############################################
# 11) Text-to-SQL Prompt
############################################

DEFAULT_TEXT_TO_SQL_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Given a question, create a correct {dialect} SQL query using ONLY the schema below. "
    "Then imagine we run the query and get results, and provide a final answer. "
    "Use this format (each step on its own line):\n\n"
    "Question: <question>\n"
    "SQLQuery: <your query>\n"
    "SQLResult: <hypothetical result>\n"
    "Answer: <final answer>\n\n"
    "Only use columns and tables listed below:\n"
    "{schema}\n\n"
    "Question: {query_str}\n"
    "Do NOT repeat these instructions.\n"
    "SQLQuery:"
)

DEFAULT_TEXT_TO_SQL_PROMPT = PromptTemplate(
    DEFAULT_TEXT_TO_SQL_TMPL, prompt_type=PromptType.TEXT_TO_SQL
)


############################################
# 12) Text-to-SQL with PGVector
############################################

DEFAULT_TEXT_TO_SQL_PGVECTOR_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Given a question, create a correct {dialect} SQL query using ONLY the schema below, "
    "plus PGVector syntax (`<->`) for nearest-neighbor search if needed. "
    "Then, imagine we run that query and provide a final answer. "
    "Use this format (each step on its own line):\n\n"
    "Question: <question>\n"
    "SQLQuery: <your query>\n"
    "SQLResult: <hypothetical result>\n"
    "Answer: <final answer>\n\n"
    "IMPORTANT: Use `[query_vector]` as a placeholder for the vector, do NOT insert actual embeddings. "
    "Only use columns/tables in the schema:\n"
    "{schema}\n\n"
    "Question: {query_str}\n"
    "Do NOT repeat these instructions.\n"
    "SQLQuery:"
)

DEFAULT_TEXT_TO_SQL_PGVECTOR_PROMPT = PromptTemplate(
    DEFAULT_TEXT_TO_SQL_PGVECTOR_TMPL, prompt_type=PromptType.TEXT_TO_SQL
)


############################################
# 13) Table Context Prompts
############################################

DEFAULT_TABLE_CONTEXT_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "We have a table schema:\n"
    "---------------------\n"
    "{schema}\n"
    "---------------------\n"
    "We also have the following context:\n"
    "{context_str}\n"
    "---------------------\n"
    "Using only this schema and context, respond to:\n"
    "{query_str}\n"
    "Do NOT repeat these instructions.\n"
    "<<ANSWER>>\n"
)

DEFAULT_TABLE_CONTEXT_PROMPT = PromptTemplate(
    DEFAULT_TABLE_CONTEXT_TMPL, prompt_type=PromptType.TABLE_CONTEXT
)


DEFAULT_TABLE_CONTEXT_QUERY = (
    "Provide a high-level description of the table and each of its columns. "
    "Format your answer as:\n"
    "TableDescription: <description>\n"
    "Column1Description: <description>\n"
    "Column2Description: <description>\n\n"
)


DEFAULT_REFINE_TABLE_CONTEXT_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Table schema:\n"
    "---------------------\n"
    "{schema}\n"
    "---------------------\n"
    "Additional context:\n"
    "{context_msg}\n"
    "---------------------\n"
    "Task: {query_str}\n"
    "Existing answer: {existing_answer}\n"
    "Refine the answer if the new context is relevant, else keep it. "
    "Do NOT repeat these instructions.\n"
    "<<ANSWER>>\n"
)

DEFAULT_REFINE_TABLE_CONTEXT_PROMPT = PromptTemplate(
    DEFAULT_REFINE_TABLE_CONTEXT_TMPL, prompt_type=PromptType.TABLE_CONTEXT
)


############################################
# 14) Knowledge-Graph Triplet Extraction
############################################

DEFAULT_KG_TRIPLET_EXTRACT_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Below is some text. Extract up to {max_knowledge_triplets} knowledge triplets in (subject, predicate, object) form. "
    "Avoid stopwords.\n"
    "---------------------\n"
    "Examples:\n"
    "Text: Alice is Bob's mother.\n"
    "Triplets:\n(Alice, is mother of, Bob)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
    "Do NOT repeat these instructions.\n"
    "<<ANSWER>>\n"
)
DEFAULT_KG_TRIPLET_EXTRACT_PROMPT = PromptTemplate(
    DEFAULT_KG_TRIPLET_EXTRACT_TMPL,
    prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT,
)


DEFAULT_DYNAMIC_EXTRACT_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Extract up to {max_knowledge_triplets} knowledge triplets from the text in (head, relation, tail) form, with types.\n"
    "---------------------\n"
    "INITIAL ONTOLOGY:\n"
    "Entity Types: {allowed_entity_types}\n"
    "Relation Types: {allowed_relation_types}\n\n"
    "You may introduce new types if necessary.\n\n"
    "Output in JSON array form: "
    "[{'head': '', 'head_type': '', 'relation': '', 'tail': '', 'tail_type': ''}, ...]\n"
    "Do NOT repeat these instructions.\n"
    "---------------------\n"
    "Text: {text}\n"
    "Output:\n"
    "<<ANSWER>>\n"
)

DEFAULT_DYNAMIC_EXTRACT_PROMPT = PromptTemplate(
    DEFAULT_DYNAMIC_EXTRACT_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)


DEFAULT_DYNAMIC_EXTRACT_PROPS_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Extract up to {max_knowledge_triplets} knowledge triplets (head, relation, tail) with types and properties.\n"
    "---------------------\n"
    "INITIAL ONTOLOGY:\n"
    "Entity Types: {allowed_entity_types}\n"
    "Entity Properties: {allowed_entity_properties}\n"
    "Relation Types: {allowed_relation_types}\n"
    "Relation Properties: {allowed_relation_properties}\n\n"
    "You may introduce new types if necessary.\n\n"
    "Output in JSON array form: "
    "[{'head': '', 'head_type': '', 'head_props': {...}, 'relation': '', 'relation_props': {...}, 'tail': '', 'tail_type': '', 'tail_props': {...}}, ...]\n"
    "Do NOT repeat these instructions.\n"
    "---------------------\n"
    "Text: {text}\n"
    "Output:\n"
    "<<ANSWER>>\n"
)

DEFAULT_DYNAMIC_EXTRACT_PROPS_PROMPT = PromptTemplate(
    DEFAULT_DYNAMIC_EXTRACT_PROPS_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)


############################################
# 15) HYDE
############################################

HYDE_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Write a passage to answer the question below, including as many key details as possible. "
    "Do NOT repeat these instructions.\n\n"
    "{context_str}\n\n"
    "<<ANSWER>>\n"
)

DEFAULT_HYDE_PROMPT = PromptTemplate(HYDE_TMPL, prompt_type=PromptType.SUMMARY)


############################################
# Simple Input
############################################

DEFAULT_SIMPLE_INPUT_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Please respond directly to this query:\n\n"
    "{query_str}\n\n"
    "Do NOT repeat these instructions.\n"
    "<<ANSWER>>\n"
)

DEFAULT_SIMPLE_INPUT_PROMPT = PromptTemplate(
    DEFAULT_SIMPLE_INPUT_TMPL, prompt_type=PromptType.SIMPLE_INPUT
)


############################################
# JSON Path
############################################

DEFAULT_JSON_PATH_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Given the JSON schema below:\n"
    "{schema}\n\n"
    "Write a valid JSONPath query to retrieve the requested data. "
    "Use the format: `JSONPath: <JSONPath>`.\n"
    "Example:\n"
    "Task: What is John's age?\n"
    "Response: JSONPath: $.John.age\n\n"
    "Task: {query_str}\n"
    "Do NOT repeat these instructions.\n"
    "Response:\n"
    "<<ANSWER>>\n"
)

DEFAULT_JSON_PATH_PROMPT = PromptTemplate(
    DEFAULT_JSON_PATH_TMPL, prompt_type=PromptType.JSON_PATH
)


############################################
# Choice Select
############################################

DEFAULT_CHOICE_SELECT_PROMPT_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "You have a list of documents with numbers and summaries. You also have a question.\n"
    "Reply with the document numbers you should consult (in order of relevance) and a relevance score 1-10. "
    "Do NOT include irrelevant docs.\n"
    "Example:\n"
    "Doc: 9, Relevance: 7\n"
    "Doc: 3, Relevance: 4\n"
    "Doc: 7, Relevance: 3\n\n"
    "Do NOT repeat these instructions.\n\n"
    "{context_str}\n"
    "Question: {query_str}\n"
    "<<ANSWER>>\n"
)

DEFAULT_CHOICE_SELECT_PROMPT = PromptTemplate(
    DEFAULT_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
)


############################################
# RankGPT Rerank Template
############################################

RANKGPT_RERANK_PROMPT_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "Search Query: {query}\n"
    "You have {num} passages. Rank them by relevance to the query in descending order using [#] notation. "
    "Example: [1] > [2]. Output ONLY the ordering, no explanation.\n"
    "Do NOT repeat these instructions.\n"
    "<<ANSWER>>\n"
)

RANKGPT_RERANK_PROMPT = PromptTemplate(
    RANKGPT_RERANK_PROMPT_TMPL, prompt_type=PromptType.RANKGPT_RERANK
)


############################################
# JSONalyze Query Template
############################################

DEFAULT_JSONALYZE_PROMPT_TMPL = (
    "<<INSTRUCTIONS>>\n"
    "You have a table named '{table_name}' with this schema:\n"
    "{table_schema}\n\n"
    "Generate a valid SQLite SQL query that answers the question:\n"
    "{question}\n\n"
    "Format:\n"
    "SQLQuery: <query>\n"
    "Do NOT repeat these instructions.\n"
    "<<ANSWER>>\n"
)

DEFAULT_JSONALYZE_PROMPT = PromptTemplate(
    DEFAULT_JSONALYZE_PROMPT_TMPL, prompt_type=PromptType.TEXT_TO_SQL
)
