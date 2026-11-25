# ResearchMind

ResearchMind is a Streamlit research assistant with a command-line runner. It searches with Tavily, reads a selected page, drafts a report with Google Gemini, and generates a critique.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key
- A Tavily API key

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a local `.env` from the example and add your API keys:

```bash
cp .env.example .env
```

Set `GEMINI_API_KEY`, `GEMINI_MODEL` (default `gemini-3.5-flash`), and `TAVILY_API_KEY` in `.env`. Keep this file private; it is ignored by Git.

For Streamlit Community Cloud, add the same values under **App settings → Secrets** using TOML syntax:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
GEMINI_MODEL = "gemini-3.5-flash"
TAVILY_API_KEY = "your_tavily_api_key"
```

## Run the web app

```bash
streamlit run app.py
```

## Run the command-line version

```bash
python pipeline.py
```

Both entry points make live calls to Google Gemini and Tavily and require valid credentials and network access. Generated reports are drafts; verify sources and claims before relying on them.
