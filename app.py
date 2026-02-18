import os
import sys
import functools
import re
from typing import Annotated, Literal, TypedDict

# Configuración de codificación para evitar errores en entornos Windows/Docker
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

# --- DEFINICIÓN DEL ESTADO Y AGENTES ---

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

search_template = """Your job is to search the web for related news that would be relevant to generate the article described by the user.
Do not write the article. Use Tavily search and return a plain-text summary of the key findings."""

outliner_template = """Take the search findings and the user's topic to generate a logical outline for a news article."""

writer_template = """You are a professional journalist. Write a complete article in ENGLISH based on the outline.
You MUST use this exact format:

TITLE: <write the title here>
BODY: <write the full article here using markdown for subheaders if needed>

Do not use bolding on the labels TITLE: or BODY:. Always write in ENGLISH."""

# --- NODOS Y LÓGICA DEL GRAFO ---

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {'messages': [result]}

def should_search(state) -> Literal["tools", "outliner"]:
    messages = state['messages']
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "outliner"

def message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text").strip()
    return str(content).strip()

@st.cache_resource(show_spinner=False)
def build_graph(gemini_api_key: str, tavily_api_key: str):
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)
    tools = [TavilySearchResults(max_results=5)]

    search_node = functools.partial(agent_node, agent=create_agent(llm, tools, search_template), name="Search")
    outliner_node = functools.partial(agent_node, agent=create_agent(llm, [], outliner_template), name="Outliner")
    writer_node = functools.partial(agent_node, agent=create_agent(llm, [], writer_template), name="Writer")

    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("outliner", outliner_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("search")
    workflow.add_conditional_edges("search", should_search)
    workflow.add_edge("tools", "search")
    workflow.add_edge("outliner", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()

# --- INTERFAZ STREAMLIT ---

def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def main():
    st.set_page_config(page_title="AI News Writer", page_icon="📰", layout="centered")
    local_css("style.css")

    st.title("AI News Writer Agent")
    
    with st.sidebar:
        st.header("API Configuration")
        gemini_api_key = st.text_input("Gemini API Key", type="password")
        tavily_api_key = st.text_input("Tavily API Key", type="password")
        st.divider()
        if st.button("Reset App", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if not gemini_api_key.strip() or not tavily_api_key.strip():
        st.info("Please enter your API Keys in the sidebar to start.")
        st.stop()
    
    user_prompt = st.text_area(
        "What should the article be about?", 
        placeholder="e.g., The impact of renewable energy in 2026...",
        height=150
    )

    if st.button("Generate Article", type="primary"):
        if not user_prompt.strip():
            st.warning("Please enter a topic.")
            return

        try:
            with st.spinner("Searching and writing..."):
                graph = build_graph(gemini_api_key.strip(), tavily_api_key.strip())
                result = graph.invoke(
                    {"messages": [HumanMessage(content=user_prompt.strip())]},
                    config={"recursion_limit": 50},
                )

            # Buscar el último mensaje del Writer que contenga la estructura esperada
            output_text = ""
            for msg in reversed(result.get("messages", [])):
                text = message_text(msg)
                if "TITLE:" in text.upper() or "BODY:" in text.upper():
                    output_text = text
                    break

            if output_text:
                st.divider()
                
                # Procesamiento robusto del Título y Cuerpo
                # Usamos regex para ignorar mayúsculas/minúsculas y asteriscos de markdown
                title_match = re.search(r'(?i)TITLE:\s*(.*)', output_text)
                body_match = re.search(r'(?i)BODY:\s*([\s\S]*)', output_text)

                if title_match and body_match:
                    title_content = title_match.group(1).split('\n')[0].replace('**', '').strip()
                    body_content = body_match.group(1).strip()
                    
                    st.markdown(f"# {title_content}")
                    st.markdown(body_content)
                else:
                    # Fallback si el regex falla pero hay texto
                    st.markdown(output_text)
            else:
                st.error("Could not extract article. Try a different prompt.")
                    
        except Exception as exc:
            st.error(f"Error: {exc}")

if __name__ == "__main__":
    main()