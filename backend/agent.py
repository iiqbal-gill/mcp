import os
import re
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Paths
DB_PATH = os.path.join(os.path.dirname(__file__), "../data/vector_db")

# 1. OPTIMIZATION: Reduce Retrieval Count (k=2 is much faster than k=5)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) 

# 2. Define the Tool Function
def search_prospectus(query: str) -> str:
    """Searches the UET Prospectus for relevant info."""
    print(f"   ⚡ Searching: '{query}'...")
    try:
        docs = retriever.invoke(query)
        if not docs:
            return "No info found."
        
        # OPTIMIZATION: Limit context size (400 chars per doc) to speed up reading
        # This drastically reduces the "thinking" time for the final answer.
        context = "\n".join([f"- {doc.page_content[:450]}..." for doc in docs])
        return context
    except Exception as e:
        return "Error."

# 3. Initialize LLM
llm = ChatOllama(model="gemma3", temperature=0)

# --- TURBO AGENT ENGINE ---
def run_manual_agent(user_query: str):
    
    # OPTIMIZATION: Short, punchy system prompt to save token processing time
    system_prompt = f"""
    You are the UET AI Agent. Protocol:
    1. If the user asks about UET facts, you MUST search. Output: Action: Search [query]
    2. If it's just "Hi", reply normally.
    
    User: {user_query}
    """

    # --- Turn 1: Decision (Fast) ---
    response_1 = llm.invoke(system_prompt).content
    
    # Regex to catch the search command
    match = re.search(r"Action:\s*Search\s*\[?\"?'?(.*?)\]?\"?'?$", response_1, re.IGNORECASE | re.MULTILINE)
    
    if match:
        search_term = match.group(1).strip().strip('"').strip("'").strip("[]")
        
        # --- Turn 2: Execution (Fast) ---
        observation = search_prospectus(search_term)
        
        # --- Turn 3: Final Answer (Optimized) ---
        final_prompt = f"""
        User: {user_query}
        Context: {observation}
        Answer briefly using the context.
        """
        return llm.invoke(final_prompt).content
    
    # Fallback: If it refused to search but gave a generic "I can't" message, force search.
    if "sorry" in response_1.lower() or "model" in response_1.lower():
        observation = search_prospectus(user_query)
        return llm.invoke(f"Context: {observation}\nUser: {user_query}\nAnswer:").content

    return response_1

# --- GUARDRAIL ---
def is_department_related(query: str) -> bool:
    keywords = [
        "department", "course", "admission", "faculty", "fee", "program", 
        "engineering", "syllabus", "professor", "phd", "msc", "uet", 
        "computer", "civil", "electrical", "mechanical", "scholarship",
        "dean", "merit", "campus", "hostel", "library", "sport", "fee"
    ]
    return any(k in query.lower() for k in keywords)

# --- MAIN ENTRY POINT ---
def process_query(user_query: str):
    if not is_department_related(user_query):
        return "I only answer department information."
    
    try:
        return run_manual_agent(user_query)
    except Exception as e:
        return f"Error: {str(e)}"