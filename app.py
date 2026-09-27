# DDS Enterprise HR Chatbot
# Render-ready version (originally built for Hugging Face Spaces)

import logging
import os
import sys
import time

import gradio as gr
from pinecone import Pinecone, ServerlessSpec

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.pinecone import PineconeVectorStore


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("dds-hr-enterprise")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Set it under Render > Environment.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing. Set it under Render > Environment.")

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)
Settings.chunk_size = 600
Settings.chunk_overlap = 200

SYSTEM_PROMPT = """
You are AYesha, the Decoding Data Science (DDS) Enterprise HR Chatbot.

Answer questions exclusively using the latest DDS HR Handbook content supplied
to the retrieval system.

Rules:
- Only answer questions directly related to DDS HR policies in the handbook.
- Do not answer general questions unrelated to DDS HR.
- Do not provide confidential information such as salary details.
- If the answer is not supported by the handbook, do not guess.
- For confidential, unsupported, or out-of-scope requests, respond:
  "I'm sorry, I can only answer questions about the latest DDS HR policies.
  For confidential or other queries, please email
  connect@decodingdatascience.com."
- Keep responses concise, professional, and easy to understand.
- Base the final answer only on retrieved
