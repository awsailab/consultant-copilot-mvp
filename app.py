import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Consultant's Copilot MVP", layout="wide")
load_dotenv()

# Check for API key
if "OPENAI_API_KEY" not in os.environ:
    st.error("OPENAI_API_KEY not found. Please set it in your .env file.")
    st.stop()

# --- RAG/VECTOR STORE SETUP ---
# Initialize the OpenAI Embeddings
embedding_function = OpenAIEmbeddings()
db_path = "chroma_db"

# Check if the database directory exists
if not os.path.exists(db_path):
    st.error(f"ChromaDB database not found at {db_path}. Please run `python ingest.py` first.")
    st.stop()

# Load the vector store
vector_store = Chroma(
    persist_directory=db_path, 
    embedding_function=embedding_function,
    collection_name="consultant_copilot"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Custom RAG Tool for CrewAI
class RAGTool:
    def __init__(self, retriever):
        self.retriever = retriever

    def search(self, query: str) -> str:
        """Searches the knowledge base for relevant context."""
        docs = self.retriever.get_relevant_documents(query)
        context = "\n---\n".join([doc.page_content for doc in docs])
        return f"Relevant Context:\n{context}"

    def __str__(self):
        return "RAGTool"

    def __repr__(self):
        return "RAGTool(retriever)"

# Instantiate the custom tool
rag_tool = RAGTool(retriever).search

# --- CREWAI AGENTS & TASKS ---

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# 1. Analyst Agent
analyst = Agent(
    role='Lead Business Analyst',
    goal='Extract client pain points, business goals, and technical constraints from a meeting transcript.',
    backstory='You are a master analyst. Your job is to listen to a client conversation and pull out all the critical requirements, constraints, and opportunities.',
    verbose=True,
    llm=llm,
    allow_delegation=False
)

# 2. Architect Agent (with RAG tool)
architect = Agent(
    role='Senior AWS Solutions Architect',
    goal='Create a compelling, sales-oriented proposal for an AWS migration and AI readiness assessment, using relevant context from our knowledge base.',
    backstory=(
        "You are 'Solution Architect', the lead AI and Cloud strategist. You are a confident expert and a persuasive communicator. "
        "Your goal is to convert a prospect into a client. "
        "Use a confident, expert, and forward-looking tone. Frame technical steps in terms of business value. "
        "Always end with a clear, confident call to action for a 'Phase 1: Discovery & AI Readiness Assessment'."
    ),
    verbose=True,
    llm=llm,
    tools=[rag_tool],  # <-- This agent can use your knowledge base
    allow_delegation=False
)

# 1. Analyst Task
task_analyze = Task(
    description='Analyze the provided meeting transcript. Identify and list the client\'s primary pain points, their stated business goals, and any mentioned technical constraints (e.g., databases, existing tech).',
    expected_output='A structured markdown report with three sections: 1. Pain Points, 2. Business Goals, 3. Technical Constraints.',
    agent=analyst
)

# 2. Architect Task
task_propose = Task(
    description=(
        'Using the analyst\'s report and our knowledge base, draft a confident, sales-oriented solution proposal. '
        'Start by acknowledging their pain points. '
        'Tie our "AWS Migration" and "AI Readiness" services to their goals. '
        'Use the `RAGTool` with queries like "AWS migration case study" or "AI readiness checklist" to find supporting evidence. '
        'Conclude with a strong call to action for a Phase 1 engagement.'
    ),
    expected_output='A persuasive, client-ready proposal document in markdown format.',
    agent=architect,
    context=[task_analyze]  # This task depends on the output of the first task
)

# Create the Crew
consulting_crew = Crew(
    agents=[analyst, architect],
    tasks=[task_analyze, task_propose],
    process=Process.sequential,
    verbose=2
)

# --- STREAMLIT UI ---
st.title("Consultant's Copilot MVP 🚀")
st.subheader("AWS Readiness & Legacy Migration Advisor")

st.markdown("""
Upload a client meeting transcript (.txt file) to generate a full analysis and draft proposal.
""")

uploaded_file = st.file_uploader("Upload your transcript (.txt)", type="txt")

if uploaded_file is not None:
    # Read the transcript
    transcript = uploaded_file.read().decode("utf-8")
    
    st.info("Transcript uploaded. Kicking off the agent crew... 🤖")
    
    # Run the crew
    with st.spinner("Agents are analyzing and drafting the proposal..."):
        try:
            # The input is a dictionary
            inputs = {'transcript': transcript}
            result = consulting_crew.kickoff(inputs=inputs)
            
            # Display the final result
            st.success("Proposal Generated!")
            st.markdown(result)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please ensure your OPENAI_API_KEY is correct and has funds.")