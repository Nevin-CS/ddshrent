# DDS AskHR AI

An enterprise HR policy assistant built with **Gradio**, **LlamaIndex**,
and **OpenAI** (GPT-4o mini + text-embedding-3-small). Answers employee
questions using a retrieval-augmented pipeline grounded in your HR
policy PDFs — with voice input, multilingual answers, live PDF
uploads, feedback logging, and conversation export.

> Educational / portfolio project. The included documents are
> synthetic examples, not real company policy, and this app does not
> constitute legal advice.

## Features

- **RAG over your HR PDFs** — answers are grounded in the documents
  in `data/`, not the model's general knowledge
- **Source citations with relevance scores** on every answer
- **Conversation memory** within a session
- **Multilingual answers** — pick a language from the dropdown
- **Voice input** — ask by microphone, transcribed via Whisper
- **Live knowledge-base expansion** — upload new PDFs through the UI,
  no code changes or redeploy needed
- **Feedback logging** — 👍/👎 on any answer, logged to
  `feedback/feedback_log.csv`
- **Conversation export** — download the chat as a Markdown file
- **Responsible AI guardrails** — the system prompt refuses to
  invent policy details, flags conflicting documents instead of
  guessing, and redirects personal/case-specific questions to HR

## Project structure

```
DDS-AskHR-AI/
├── app.py              # Main application (Gradio + LlamaIndex)
├── requirements.txt
├── render.yaml          # One-click Render blueprint
├── .gitignore
├── README.md
└── data/                # Your HR policy PDFs go here
```

`uploads/` and `feedback/` are created automatically at runtime and
are git-ignored — no need to create them by hand.

## Setup

1. Clone or download this repo.
2. Put your HR policy PDFs in `data/` (any filenames — the app loads
   every `.pdf` it finds there automatically, no hardcoding).
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set your OpenAI key:
   ```
   export OPENAI_API_KEY=sk-...
   ```
5. Run it:
   ```
   python app.py
   ```

## Deploying

The same `app.py` runs unmodified on both platforms below — no
platform-specific branches or separate versions to maintain.

### Hugging Face Spaces

1. Create a new Space (SDK: Gradio).
2. Push/upload `app.py`, `requirements.txt`, and the `data/` folder.
3. Under **Settings → Variables and secrets**, add `OPENAI_API_KEY`
   as a secret.
4. The Space builds and launches automatically.

### Render

1. Push this repo to GitHub.
2. On Render: **New → Web Service** → connect the repo.
3. Build command: `pip install -r requirements.txt`
   Start command: `python app.py`
4. Add `OPENAI_API_KEY` under **Environment**.
5. Deploy — Render auto-detects the `render.yaml` blueprint if you
   use "New → Blueprint" instead, which sets the build/start commands
   for you.

Render assigns a dynamic `$PORT`; `app.py` reads it automatically and
falls back to `7860` locally / on Hugging Face.

## Notes on the free tiers

- **Render free/Starter** and **Hugging Face free Spaces** both have
  limited memory. The first request after a cold start can be slow
  while the model libraries load — this is expected.
- Files added via the in-app PDF upload, and the feedback log, live
  on local disk and are **not persisted** across a redeploy/restart on
  either platform's free tier. For permanent storage, put source PDFs
  in `data/` and commit them, or wire up external storage.

## License

Educational/demo project — adapt freely for your own use.
