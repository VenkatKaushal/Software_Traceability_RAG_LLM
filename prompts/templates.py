from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType


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


CLASS_TRACE_TEMPLATE = \
"""
What are the names of the classes that are related to the following use case requirement?
{requirement}

Provide the answer in a list format and provide ONLY the list of class names as a JSON list.
[<"Class 1 Name">, <"Class 2 Name">, ... <"Class N Name">] where N can be up to 10.
"""


DOC_STRUCTURE_TEMPLATE = \
"""
Document Structure:
Class Name: <Class Name>
Attributes: 
<Attribute1 Name>: <Attribute1 Type>
<Attribute2 Name>: <Attribute2 Type>
...
<AttributeN Name>: <AttributeN Type>

<Class Docstring>

Methods:

Method Name: <Method1 Name>
Signature: <Method1 Signature>
Class Name: <Class Name>
Docstring:
<Method1 Docstring>

Calls:
<Method1 Name> calls <Method Name>

Called By:
<Method1 Name> called by <Method Name>
"""


CODE_EXAMPLE = \
"""
    "Example:"
    "Doc Text: "
    ""
    "Class Name: AdvertisementAgencyManager"
    "Attributes: "
    "serialVersionUID: long"
    "dbNews: IDBNews"
    "/**"
    " * Implementing the management advertisement"
    " * For the operator agency. Contains methods for managing"
    " * News."
    " *"
    " */"
    "Method Name: clearNews"
    "Signature: AdvertisementAgencyManager.clearNews(int)"
    "Class Name: AdvertisementAgencyManager"
    "Docstring: "
    "/**"
    "     * Method which removes news from the database. Uses the (@Link Boolean"
    "     * unisa.gps.etour.repository.IDBNews # clearNews (int))"
    "     *"
    "     * @Param id pNewsID news be erased."
    "     * @Return true if the clearing was successful or FALSE otherwise."
    "     * @Throws RemoteException"
    "     *"
    "     */"
    "Method Name: modifyNews"
    "Calls: "
    "modifyNews calls IDBNews.modifyNews(BeanNews)"
    "modifyNews calls ControlData.checkBeanNews(BeanNews)"
    ""
"""


METHOD_CALL_EXTRACTION_TEMPLATE = \
"Also, provide the purpose of each method call where a method 'calls' or 'called by' another method" \
+ "Provide the summary in the following format:"\
+ "<Method Call Summary>\n"\
+ "<Method Call Summary>\n"\
+ "...\n\n"\
+ "<Method Called By Summary>\n"\
+ "<Method Called By Summary>\n"\
+ "...\n\n"\


CODE_SUMMARY_EXTRACT_TEMPLATE = \
"You are an expert in summarizing Code to the relevant Use case scenario where it can be used." \
+ "Given a class in a Java code file, generate a class summary to map a given use case requirements to the given java code." \
+ "The summary should capture the purpose of the class and its attributes and methods"\
+ "Given a doc with the following structure that will be useful for traceability where we map use case case requirements with code elements:\n"\
+ f"{DOC_STRUCTURE_TEMPLATE}"\
+ f"{CODE_EXAMPLE}"\
+ "Provide the summary in simple abstract use case scenario language in the following format: "\
+ "Class Name: <Class Summary>\n"\
+ "Method Name: <Method Summary>\n"\
+ "Method Name: <Method Summary>\n"\
+ "...\n"\
+ " Given the following code, provide the purpose of the class and each of the present methods. "\
+ "{method_calls_str}"\


def code_summarization_template(use_method_calls=False):
    if use_method_calls:
        template = CODE_SUMMARY_EXTRACT_TEMPLATE.format(
            method_calls_str=METHOD_CALL_EXTRACTION_TEMPLATE
        )
    else:
        template = CODE_SUMMARY_EXTRACT_TEMPLATE.format(method_calls_str="")
    
    template += " Here is the given code:\n {context_str}\n\n"\
    
    return template



CODE_KG_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    f"Given a doc with the following structure:\n{DOC_STRUCTURE_TEMPLATE}"
    "---------------------\n"
    "Extract the upto "
    "{max_knowledge_triplets} "
    "triples from the code text such that triples contain class names and function calls and parameters.\n"
    "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
    "Subject is the class name"
    "Predicate is the function name or the attribute name"
    "Object is the attribute type in case of attribute predicate"
    "and all the parameter types in case of a function\n"
    "Also directly extract the calls and called by triplets as in below example"
    "---------------------\n"
    f"{CODE_EXAMPLE}"
    "Triplets:"
    "(AdvertisementAgencyManager, serialVersionUID, int)"
    "(AdvertisementAgencyManager, dbNews, IDBNews)"
    "(AdvertisementAgencyManager, clearNews, int)"
    "(modifyNews, IDBNews.modifyNews, BeanNews)"
    "(modifyNews, ControlData.checkBeanNews, BeanNews)"

    "---------------------\n"
    "Code: {text}\n"
    "Triplets:\n"
)


CODE_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "Given a doc extract the upto {keywords} that will be useful for traceability where we map use case case requirements with code elements."
    "For example extract out the key code elements like class names, function names, variable names, etc. from the code text."
    "Avoid stopwords."
    f"{CODE_EXAMPLE}"
    "Keywords:"
    "AdvertisementAgencyManager"
    "serialVersionUID"
    "AdvertisementAgencyManager"
    "dbNews"
    "IDBNews"
    "clearNews"
    "modifyNews"
    "BeanNews"
    "ControlData"
    "checkBeanNews"
    "---------------------\n"
    "Code: {context_str}\n"
)

CODE_KG_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
    CODE_KG_KEYWORD_EXTRACT_TEMPLATE_TMPL,
    prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT,
)


CODE_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
    CODE_KEYWORD_EXTRACT_TEMPLATE_TMPL, prompt_type=PromptType.KEYWORD_EXTRACT
)


REQ2CODE_QA_TEMPLATE = \
"Given a doc with the following structure that will be useful for traceability where we map use case case requirements with code elements:\n"\
+ f"{DOC_STRUCTURE_TEMPLATE}"\
+ f"{CODE_EXAMPLE}"\
+ " Here is the context:"\
+ "{context_str}\n"\
+ " Given the contextual information,"\
+ " your task is to generate {num_questions_per_chunk} questions this context can provide"\
+ " specific answers related to requirements to Java Class traceability."\
+ " to which are unlikely to be found elsewhere. The questions should be diverse in nature across the document."\
+ " Restrict the questions to the context information provided."\
+ " Provide the questions in a list format where each question is a string in the list."




SCENARIO_GEN_TEMPLATE = (
    "You are an expert in mapping Code to Use case scenario and you generate multiple similar scenario some"
    " generic abstract scenarios related to the functionality in the scenario and half concrete based on a "
    "single input query."
    "Generate {num_queries} requirements, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)