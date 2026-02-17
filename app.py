import os
import sys
import functools
from typing import Annotated, Literal, TypedDict

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

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_message}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    if tools:
        return prompt | llm.bind_tools(tools)
    else:
        return prompt | llm

search_template = """Your job is to search the web for related news that would be relevant to generate the article described by the user.
NOTE: Do not write the article.
Use Tavily search if needed, but call tools at most once.
Then return a plain-text summary of the key findings for the outliner node."""

outliner_template = """Your job is to take as input a list of articles from the web along with users instruction on what article they want to write and generate an outline for the article."""

writer_template = """Your job is to write an article in ENGLISH. You must do it in this EXACT format, without using markdown (like ** or #) on the labels:

TITLE: <write the title here>
BODY: <write the full article here>

NOTE: Do not just copy the outline. You need to write a complete, well-structured article with the info provided by the outline. Always write in ENGLISH."""

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
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(part for part in text_parts if part).strip()
    return str(content).strip()

@st.cache_resource(show_spinner=False)
def build_graph(gemini_api_key: str, tavily_api_key: str):
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key)
    tools = [TavilySearchResults(max_results=5)]

    search_agent = create_agent(llm, tools, search_template)
    outliner_agent = create_agent(llm, [], outliner_template)
    writer_agent = create_agent(llm, [], writer_template)

    search_node = functools.partial(agent_node, agent=search_agent, name="Search Agent")
    outliner_node = functools.partial(agent_node, agent=outliner_agent, name="Outliner Agent")
    writer_node = functools.partial(agent_node, agent=writer_agent, name="Writer Agent")

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
        st.markdown("Get your keys at [Google AI Studio](https://aistudio.google.com/) and [Tavily](https://tavily.com/).")
        
        st.divider()
        if st.button("Reset App", use_container_width=True):
            st.rerun()

    # --- PANTALLA DE INICIO ---
    if not gemini_api_key.strip() or not tavily_api_key.strip():
        st.markdown("""
        This agent is an automated research and writing assistant powered by **LangGraph**. Give it a topic, and it will autonomously search the web, draft an outline, and write a complete, structured news article.
        
        **How the workflow operates:**
        * **Search Node:** Uses Tavily to scour the web for the latest, most relevant news on your topic.
        * **Outliner Node:** Processes the search results to create a well-structured article outline.
        * **Writer Node:** Drafts the final, full-length article based on the outline.

        **To get started:**
        Please enter both your **Gemini** and **Tavily API Keys** in the sidebar.
        """)

    st.markdown("This agent searches the web, creates an outline, and writes a complete news article in English.")
    user_prompt = st.text_area("What should the article be about?", placeholder="e.g., The latest trends in Artificial Intelligence in 2026...")

    if st.button("Generate Article"):
        if not user_prompt.strip():
            st.warning("Please enter a topic for the article.")
            return

        try:
            with st.spinner("Researching and writing (this may take a few seconds)..."):
                graph = build_graph(gemini_api_key.strip(), tavily_api_key.strip())
                result = graph.invoke(
                    {"messages": [HumanMessage(content=user_prompt.strip())]},
                    config={"recursion_limit": 50},
                )

            messages = result.get("messages", [])
            output_text = ""
            
            for msg in reversed(messages):
                output_text = message_text(msg)
                if output_text and ("TITLE" in output_text.upper() or "BODY" in output_text.upper()):
                    break

            if output_text:
                clean_text = output_text.replace("**TITLE:**", "TITLE:").replace("**TITLE**:", "TITLE:")
                clean_text = clean_text.replace("**BODY:**", "BODY:").replace("**BODY**:", "BODY:")
                
                if "BODY:" in clean_text:
                    parts = clean_text.split("BODY:", 1)
                    title_part = parts[0].replace("TITLE:", "").strip()
                    body_part = parts[1].strip()
                    
                    st.success("Article completed!")
                    st.divider()
                    st.header(title_part)
                    st.markdown(body_part)
                else:
                    st.success("Article completed!")
                    st.divider()
                    st.markdown(clean_text)
            else:
                st.error("The model returned an empty or unstructured response. Please try again.")
                with st.expander("View debug messages"):
                    st.write(messages)
                    
        except Exception as exc:
            st.error(f"An error occurred: {exc}")

if __name__ == "__main__":
    main()