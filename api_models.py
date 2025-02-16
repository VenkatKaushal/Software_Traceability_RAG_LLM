"""
This script configures and utilizes a custom LLM (Large Language Model) class using HuggingFace's transformer models. It provides the functionality to set up both embedding models and LLMs, generate predictions, and handle streaming and asynchronous operations.

Functions and Classes:

1. set_embed_model:
   - Sets the embedding model for use, defaulting to "BAAI/bge-small-en-v1.5". This model is used for embedding text.

2. set_llm:
   - Configures the LLM based on the given type (e.g., "deepseek", "llama2", "mistral"). If no specific LLM type is provided, a default LLM is set. This function does not require API keys for the LLM configuration.

3. set_llm_and_embed:
   - A convenience function that sets both the LLM and embedding model. It uses default settings if no specific model names are provided.

4. LLM (Main Class):
   - This class defines the core LLM used in the script, extending from the `LLM` class. It handles the initialization of model and tokenizer, setting device types (CPU or CUDA), and providing methods for generating completions and chatting.
   - It includes:
     - `metadata`: Returns metadata for the LLM such as the model name and max tokens.
     - `chat`: Processes chat messages and returns a `ChatResponse`.
     - `complete`: Processes a prompt and generates a completion response.
     - `predict`: Runs the model on a formatted prompt and generates the output text.
     - `stream_chat` and `stream_complete`: These methods return responses as generators for streaming output.
     - `achat` and `acomplete`: Asynchronous methods to handle chat and completion tasks.
     - `astream_chat` and `astream_complete`: Asynchronous generators that stream chat and completion responses.

Dependencies:
- llama_index: Used for working with various indexing, LLM, and prompt templates.
- transformers: For model handling with HuggingFace's transformers library.
- torch: For interacting with models on either CPU or GPU.
- huggingface_hub: To enable login and access HuggingFace models.
- asyncio: Used for asynchronous operations in the chat and completion streams.
"""

from llama_index import *
import subprocess
from typing import Sequence, Any, List, Generator, AsyncGenerator
from pydantic import BaseModel, Field
import asyncio

from llama_index.core.llms import ChatMessage, ChatResponse, CompletionResponse
from llama_index.core.base.llms.types import LLMMetadata, TextBlock
from llama_index.core.llms import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from llama_index.core import Settings
from llama_index.core.prompts.base import BasePromptTemplate
from transformers import RobertaForQuestionAnswering
import os
model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
token = os.getenv("HUGGINGFACE_TOKEN")

def set_embed_model(model_name: str = None) -> None:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    default_model = model_name or "BAAI/bge-small-en-v1.5"
    embed_llm = HuggingFaceEmbedding(model_name=default_model, max_length=1024)
    Settings.embed_model = embed_llm

def set_llm(llm_type: str = None, model_name: str = None, api_key=None) -> None:
    """
    Set the LLM based on a simple string key. This configuration requires no API keys.
    Acceptable llm_type values (case-insensitive): "deepseek", "llama2", "mistral, google/gemma-2-2b microsoft/graphcodebert-base
    """
    # if llm_type == "microsoft/graphcodebert-base":
    #     llm = GemmaLLM(device="cpu")
    # else:
    #     raise ValueError(f"Unsupported LLM type: {llm_type}")
    # Settings.llm = llm
    llm = LLM(device="cpu")
    Settings.llm = llm

def set_llm_and_embed(
    llm_type: str = None,
    llm_name: str = None,
    embed_model_name: str = None,
):
    if llm_name:
        set_llm(llm_name)
    else:
        set_llm()
    
    if embed_model_name:
        set_embed_model(embed_model_name)
    else:
        set_embed_model()

class LLM(LLM):
    """
    DeepSeekLLM loads an open-source DeepSeek model from HuggingFace.
    No API keys are required.
    """
    model_name: str = Field(default=model, description="Model name for DeepSeek")
    max_new_tokens: int = Field(default=256, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    device: str = "cpu" 

    def __init__(self, model_name: str = None, max_new_tokens: int = None, temperature: float = None, device: str = None):
        super().__init__()
        self.model_name = model_name or model
        self.max_new_tokens = max_new_tokens or 256
        self.temperature = temperature or 0.7
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

       
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, use_auth_token=token).to(self.device)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
            max_length=self.max_new_tokens,
            supports_streaming=True,
            supports_async=True,
        )

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = "\n".join(f"{msg.role}: {msg.content}" for msg in messages)
        response_text = self.predict(prompt, **kwargs)
        return ChatResponse(content=response_text)

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        response_text = self.predict(prompt, **kwargs)
        return CompletionResponse(text=response_text)

    def predict(self, prompt: BasePromptTemplate, **prompt_args) -> str:
        formatted_prompt = prompt.format(llm = self, **prompt_args)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        print("------------------------------------prompt started------------------------------------") 
        print("prompt: ", formatted_prompt)
        print("------------------------------------prompt ended------------------------------------")  
             
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=4000, 
            do_sample=True,
            temperature=self.temperature,
        ).to(self.device)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("------------------------------------prompt answer started------------------------------------") 
        print("generated_text: ", generated_text)
        print("------------------------------------prompt answer ended------------------------------------") 
        return generated_text

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Generator[ChatResponse, None, None]:
        yield self.chat(messages, **kwargs)

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        yield self.complete(prompt, formatted, **kwargs)

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return await asyncio.to_thread(self.complete, prompt, formatted, **kwargs)

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:
        for resp in self.stream_chat(messages, **kwargs):
            yield resp
            await asyncio.sleep(0)

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
        for resp in self.stream_complete(prompt, formatted, **kwargs):
            yield resp
            await asyncio.sleep(0)