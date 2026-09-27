# Deploying DDS Enterprise HR Chatbot to Render

This folder is a Render-ready copy of the Hugging Face Space at
https://huggingface.co/spaces/decodingdatascience/hrenterprise

## What changed vs. the Hugging Face version
- `app.py`: reads the port from the `PORT` environment variable (Render assigns
  this dynamically) instead of the hardcoded `7860`. Added `.queue()` so the
  Gradio app handles concurrent requests properly behind Render's proxy.
- `requirements.txt`: Hugging Face's Docker image installs `gradio`, `openai`
  and the Pinecone SDK automatically based on the Space's `sdk` setting in
  `README.md`. Render doesn't know about that, so this file lists every
  package `app.py` actually imports.
- `Data/`: the same four HR policy PDFs, regenerated with identical text
  content.

## Environment variables to set in Render
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`

Both are read by `app.py` via `os.getenv(...)`. Never commit real key values
into this repo — add them in Render's dashboard under
Environment > Environment Variables.

## Render service settings
- Runtime: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `python app.py`
- Instance type: at least the smallest paid tier is recommended — this
  service loads llama-index + downloads embeddings the first time it runs
  a query, which can be slow/tight on free tier memory limits.

## First request after each fresh deploy
The very first time the app runs with a brand-new Pinecone index (i.e. the
index named `quickstart` doesn't exist yet in your Pinecone account), it will
read the PDFs in `Data/`, generate embeddings via OpenAI, and upload them to
Pinecone. This happens once at process startup and can take from several
seconds to a couple of minutes depending on Pinecone/OpenAI response times.
Subsequent restarts reuse the existing Pinecone index instead of rebuilding it.
