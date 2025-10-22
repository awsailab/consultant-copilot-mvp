import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
from crewai.tools import BaseTool
import markdown
from fpdf import FPDF

# --- PDF Helper Function ---
def create_pdf(md_text):
    """
    Converts markdown text to a PDF byte stream.
    Strips non-ASCII chars (like emojis) for fpdf2 compatibility.
    """
    # Convert markdown to HTML
    md_text_ascii = md_text.encode('ascii', 'ignore').decode('ascii')
    html = markdown.markdown(md_text_ascii)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.write_html(html)
    
    # Return the PDF as raw bytes
    return bytes(pdf.output())

# --- PAGE CONFIG ---
st.set_page_config(page_title="Consultant's Copilot MVP", layout="wide")
load_dotenv()

# Check for API key
if "OPENAI_API_KEY" not in os.environ:
    st.error("OPENAI_API_KEY not found. Please set it in your .env file.")
    st.stop()

# --- RAG/VECTOR STORE SETUP ---
embedding_function = OpenAIEmbeddings()
db_path = "chroma_db"
collection_name = "consultant_copilot"

if not os.path.exists(db_path):
    st.error(f"ChromaDB database not found at {db_path}. Please run `python ingest.py` first.")
    st.stop()

vector_store = Chroma(
    persist_directory=db_path, 
    embedding_function=embedding_function,
    collection_name=collection_name
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# --- RAG Tool ---
class RAGTool(BaseTool):
    name: str = "Knowledge Base Search"
    description: str = "Searches the company's knowledge base (case studies, AI readiness checklists) for relevant context."
    retriever: any = None

    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever

    def _run(self, query: str) -> str:
        """Searches the knowledge base for relevant context."""
        docs = self.retriever.invoke(query)
        context = "\n---\n".join([doc.page_content for doc in docs])
        return f"Relevant Context:\n{context}"

rag_tool = RAGTool(retriever=retriever)

# --- CREWAI AGENTS & TASKS ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

analyst = Agent(
    role='Lead Business Analyst',
    goal='Extract client pain points, business goals, and technical constraints from a meeting transcript.',
    backstory='You are a master analyst. Your job is to listen to a client conversation and pull out all the critical requirements, constraints, and opportunities.',
    verbose=True,
    llm=llm,
    allow_delegation=False
)

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
    tools=[rag_tool],
    allow_delegation=False
)

task_analyze = Task(
    description='Analyze the provided meeting transcript. Identify and list the client\'s primary pain points, their stated business goals, and any mentioned technical constraints (e.g., databases, existing tech).',
    expected_output='A structured markdown report with three sections: 1. Pain Points, 2. Business Goals, 3. Technical Constraints.',
    agent=analyst
)

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
    context=[task_analyze]
)

consulting_crew = Crew(
    agents=[analyst, architect],
    tasks=[task_analyze, task_propose],
    process=Process.sequential,
    verbose=True
)

# --- STREAMLIT UI ---
st.title("Consultant's Copilot MVP 🚀")
st.subheader("AWS Readiness & Legacy Migration Advisor")

st.markdown("""
Upload a client meeting transcript (.txt file) to generate a full analysis and draft proposal.
""")

uploaded_file = st.file_uploader("Upload your transcript (.txt)", type="txt")

if uploaded_file is not None:
    transcript = uploaded_file.read().decode("utf-8")
    
    st.info("Transcript uploaded. Kicking off the agent crew... 🤖")
    
    with st.spinner("Agents are analyzing and drafting the proposal..."):
        try:
            inputs = {'transcript': transcript}
            result_object = consulting_crew.kickoff(inputs=inputs)
            result = result_object.raw # <-- THIS IS THE FIX 
            st.success("Proposal Generated!")
            st.markdown(result)
            
            # --- START: UPDATED DOWNLOAD BUTTONS ---
            st.markdown("---") # Add a separator
            
            # Generate PDF bytes in memory
            pdf_bytes = create_pdf(result)

            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="⬇️ Download .md",
                    data=result,
                    file_name="proposal.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col2:
                st.download_button(
                    label="⬇️ Download .txt",
                    data=result,
                    file_name="proposal.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col3:
                st.download_button(
                    label="⬇️ Download .pdf",
                    data=pdf_bytes,
                    file_name="proposal.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            # --- END: UPDATED DOWNLOAD BUTTONS ---
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please ensure your OPENAI_API_KEY is correct and has funds.")
