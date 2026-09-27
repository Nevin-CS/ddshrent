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
- Base the final answer only on retrieved handbook information.
"""

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "quickstart"
DIMENSION = 1536

existing_indexes = [idx["name"] for idx in pc.list_indexes()]
index_was_created = INDEX_NAME not in existing_indexes

if index_was_created:
    logger.info("Creating Pinecone index '%s'...", INDEX_NAME)
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    time.sleep(5)
else:
    logger.info("Using existing Pinecone index '%s' — NOT deleting/rebuilding it.", INDEX_NAME)

pinecone_index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

if index_was_created:
    logger.info("Loading PDFs from the repo root and creating embeddings...")

    documents = SimpleDirectoryReader(
        input_dir=".",
        required_exts=[".pdf"],
        file_extractor={".pdf": PDFReader()},
    ).load_data()

    if not documents:
        raise ValueError("No PDF documents were found at the repo root.")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    logger.info("DDS HR handbook indexed successfully.")
else:
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_engine = index.as_query_engine(system_prompt=SYSTEM_PROMPT, similarity_top_k=4)


def query_doc(prompt: str) -> str:
    if not prompt or not prompt.strip():
        return "Please enter an HR policy question."
    try:
        response = query_engine.query(prompt.strip())
        return str(response)
    except Exception:
        logger.exception("Query error")
        return "Sorry, I couldn't process your question right now. Please try again."


demo = gr.Interface(
    fn=query_doc,
    inputs=gr.Textbox(
        label="Ask a question about DDS HR policies",
        placeholder="Example: What is the annual leave policy?",
    ),
    outputs=gr.Textbox(label="Answer"),
    title="DDS Enterprise HR Chatbot",
    description="Ask questions based on the latest DDS HR Handbook.",
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
