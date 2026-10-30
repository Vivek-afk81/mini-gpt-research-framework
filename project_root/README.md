# LexiFlow

LexiFlow converts Romanized Hinglish student messages into formal English, classifies the request, retrieves relevant sample college policies, remembers demo-user requests, and drafts a cited email.

## Local MVP

The local backend needs no Groq key:

```powershell
$env:PYTHONPATH = "."
..\venv\Scripts\python.exe main.py "meri attendance 67% hai, kya mujhe condonation mil sakta hai"
```

The local translator uses the supplied examples as a deterministic fallback. It is suitable for testing the full pipeline, not a replacement for a trained LLM.

## Optional Groq backend

Install the optional client if it is not already installed:

```powershell
..\venv\Scripts\python.exe -m pip install groq
```

Set the key for the current PowerShell session:

```powershell
$env:GROQ_API_KEY = "your-key"
$env:TRANSLATOR_BACKEND = "groq"
```

Run the same pipeline:

```powershell
..\venv\Scripts\python.exe main.py `
  "meri attendance 67% hai, kya mujhe condonation mil sakta hai" `
  --backend groq `
  --user-id demo_user `
  --name "Demo Student"
```

Groq is used for translation/classification and grounded email drafting. Responses are cached in `.cache/groq.jsonl`; the cache avoids repeating identical requests. If the key is missing, the package is unavailable, the request fails, or the response is invalid, LexiFlow falls back to the local translator and deterministic draft.

Available configuration is documented in `.env.example`. The code does not automatically read `.env`, so export variables in the shell or load them with your preferred environment manager.

## Data preparation

```powershell
..\venv\Scripts\python.exe -m src.lexiflow.data.split data\campus_messages_v2.jsonl
..\venv\Scripts\python.exe -m src.lexiflow.data.chat_format
```

These commands only clean, split, and convert the data. They do not train a model.

## Gradio app

Install Gradio, then run from `project_root`:

```powershell
..\venv\Scripts\python.exe -m pip install gradio
..\venv\Scripts\python.exe app\app.py
```

Policy files under `data/policies/` are samples and must be replaced with approved college documents before real use.