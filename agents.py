import os
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url 
from dotenv import load_dotenv

load_dotenv()

def _setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default

def extract_text(content) -> str:
    """Safely extract plain string text from any LangChain / Gemini response format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)

# Model setup
api_key = (_setting("GEMINI_API_KEY") or _setting("GOOGLE_API_KEY")).strip().strip('"').strip("'")
model_name = _setting("GEMINI_MODEL", "gemini-3.5-flash")

llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=api_key,
    max_retries=4,
)

# 1st agent 
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )

# 2nd agent 
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )

# Writer chain 
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# Critic chain 
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
