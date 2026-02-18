import os
import sys
import functools
import re
from typing import Annotated, Literal, TypedDict

# Configuración de codificación
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- ESTADO Y AGENTES ---

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        MessagesPlaceholder(variable_name="messages"),
    ])
    prompt = prompt.partial(system_message=system_message)
    return prompt | llm.bind_tools(tools) if tools else prompt | llm

# --- PROMPTS ---

search_template = """Your job is to search the web for related news relevant to the user's topic.
Do not write the article. Use Tavily and return a summary of findings."""

outliner_template = """Generate a professional news article outline based on the search findings."""

writer_template = """You are a professional journalist. Write a complete article in ENGLISH.
Use this EXACT format:

TITLE: <write title here>
BODY: <write full article here using markdown headers for sections>

Do not use bold on labels TITLE: or BODY:. Use only plain text for labels."""

# --- LÓGICA DEL GRAFO ---

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {'messages': [result]}

def should_search(state) -> Literal["tools", "outliner"]:
    messages = state['messages']
    last_message = messages[-1]
    return "tools" if getattr(last_message, "tool_calls", None) else "outliner"

def message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str): return content.strip()
    if isinstance(content, list):
        return "\n".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text").strip()
    return str(content).strip()

@st.cache_resource(show_spinner=False)
def build_graph(gemini_api_key: str, tavily_api_key: str):
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)
    tools = [TavilySearchResults(max_results=5)]
    
    workflow = StateGraph(AgentState)
    workflow.add_node("search", functools.partial(agent_node, agent=create_agent(llm, tools, search_template), name="Search"))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("outliner", functools.partial(agent_node, agent=create_agent(llm, [], outliner_template), name="Outliner"))
    workflow.add_node("writer", functools.partial(agent_node, agent=create_agent(llm, [], writer_template), name="Writer"))

    workflow.set_entry_point("search")
    workflow.add_conditional_edges("search", should_search)
    workflow.add_edge("tools", "search")
    workflow.add_edge("outliner", "writer")
    workflow.add_edge("writer", END)
    return workflow.compile()

# --- INTERFAZ ---

def local_css(file_name):
    # Inyectamos un CSS base para limpiar la "barra" oscura y mejorar el texto
    st.markdown("""
        <style>
        .stMarkdown:empty { display: none; }
        [data-testid="stMarkdownContainer"] { line-height: 1.6; }
        .main .block-container { padding-bottom: 5rem; }
        </style>
    """, unsafe_allow_html=True)
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError: pass

def main():
    st.set_page_config(page_title="AI News Writer", page_icon="📰", layout="centered")
    local_css("style.css")

    st.title("AI News Writer Agent")
    
    with st.sidebar:
        st.header("API Configuration")
        gem_key = st.text_input("Gemini API Key", type="password")
        tav_key = st.text_input("Tavily API Key", type="password")
        if st.button("Reset App", use_container_width=True):
            st.rerun()

    if not gem_key.strip() or not tav_key.strip():
        st.info("Please enter your API Keys in the sidebar to start.")
        st.stop()
    
    user_prompt = st.text_area("What should the article be about?", height=120)

    if st.button("Generate Article", type="primary"):
        if not user_prompt.strip():
            st.warning("Please enter a topic.")
            return

        try:
            with st.spinner("Searching and writing..."):
                graph = build_graph(gem_key.strip(), tav_key.strip())
                result = graph.invoke({"messages": [HumanMessage(content=user_prompt.strip())]})

            output_text = ""
            for msg in reversed(result.get("messages", [])):
                text = message_text(msg)
                if "TITLE:" in text.upper():
                    output_text = text
                    break

            if output_text:
                st.divider()
                # Extracción mejorada con Regex
                title_match = re.search(r'(?i)TITLE:\s*(.*)', output_text)
                body_match = re.search(r'(?i)BODY:\s*([\s\S]*)', output_text)

                if title_match and body_match:
                    title_content = title_match.group(1).split('\n')[0].replace('**', '').strip()
                    body_content = body_match.group(1).strip()
                    
                    st.markdown(f"# {title_content}")
                    st.markdown(body_content)
                    
                    # Botón de descarga para "rellenar" el final y ser útil
                    st.download_button("Download Article", f"{title_content}\n\n{body_content}", file_name="article.txt")
                else:
                    st.markdown(output_text.strip())
            else:
                st.error("No article content found.")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()