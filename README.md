# Consultant's Copilot MVP: AWS Readiness Advisor

This project is a "Consultant's Copilot," an AI-powered assistant built to support IT/AI consulting. This MVP (Minimum Viable Product) is designed as a post-meeting analysis tool that takes a client meeting transcript, analyzes it, and generates a sales-oriented proposal for an "AWS Legacy Migration & AI Readiness Assessment."

The system uses a multi-agent (CrewAI) approach to simulate a consulting team:
1.  An **Analyst Agent** reads the transcript to extract key requirements.
2.  An **Architect Agent** uses these requirements and a custom Knowledge Base (RAG) to draft a confident, technical, and value-driven solution proposal.

## ✨ Features

* **Transcript Analysis:** Ingests raw text transcripts from client meetings.
* **Automated Requirement Extraction:** Identifies client pain points, business goals, and technical constraints.
* **Retrieval-Augmented Generation (RAG):** The Architect agent queries a local ChromaDB vector store (loaded with your case studies and service docs) to find relevant context.
* **Automated Proposal Generation:** Outputs a client-ready, sales-oriented proposal in markdown.
* **Multi-Format Download:** Allows you to instantly download the generated proposal as a `.md`, `.txt`, or `.pdf` file.
* **Simple Web UI:** Built with Streamlit for easy file uploads and report viewing.

## 🛠️ Tech Stack

* **Development Environment:** GitHub Codespaces
* **Application Framework:** Streamlit
* **AI Agent Orchestration:** CrewAI
* **LLM Provider:** OpenAI (GPT-4o)
* **Vector Store (RAG):** ChromaDB (local)
* **Document Loading:** LangChain
* **PDF Generation:** fpdf2

## 🚀 Getting Started

This project is designed to run seamlessly in GitHub Codespaces.

### Prerequisites

1.  A GitHub Account (to use Codespaces).
2.  An **OpenAI API Key** with billing set up.

### Setup & Launch

1.  **Launch Codespace:**
    * Click the green **`<> Code`** button on this repository's main page.
    * Go to the **"Codespaces"** tab.
    * Click **"Create codespace on main"**.
    * Wait 2-5 minutes for the container to build and install all dependencies (from `requirements.txt`).

2.  **Set API Key:**
    * In the Codespace terminal, rename the example `.env` file:
        ```bash
        mv .env.example .env
        ```
    * Open the `.env` file in the editor and paste your `OPENAI_API_KEY`. The file is already in `.gitignore`, so it will not be committed.

3.  **Populate Knowledge Base:**
    * Open the `/knowledge_base` folder.
    * Delete the `.txt` placeholders.
    * Upload your own company documents:
        * Past project case studies (`.pdf`, `.txt`)
        * Service descriptions
        * Internal whitepapers or checklists

4.  **Ingest Data:**
    * This step "teaches" your AI by loading your documents into the ChromaDB vector store.
    * Run this command in the Codespace terminal **one time**:
        ```bash
        python ingest.py
        ```
    * You will see a "Ingestion complete!" message.

5.  **Run the Application:**
    * In the terminal, run the Streamlit app:
        ```bash
        streamlit run app.py
        ```
    * A "Simple Browser" tab should automatically open in your Codespace, or you can click the "Ports" tab and open the URL for port `8501`.

## Usage

1.  Open the Streamlit application in your browser.
2.  Upload a client meeting transcript (e.g., `test_transcript_01.txt`).
3.  The app will show a spinner while the CrewAI agents work.
4.  The final, formatted proposal will appear on the screen, with download buttons for `.md`, `.txt`, and `.pdf` formats.

## 📈 Future Work

This MVP provides the core value. The next steps for a production system include:

* **Deploy to AWS:** Host the application on AWS App Runner or ECS.
* **Managed Vector DB:** Migrate from local ChromaDB to a managed service like Pinecone or Amazon OpenSearch Serverless.
* **Automated Ingest:** Create an S3 $\rightarrow$ Lambda pipeline to automatically update the knowledge base when new documents are added.
* **CRM Integration:** Add a final step to automatically create a new "Opportunity" or "Draft Email" in a CRM like HubSpot or Salesforce.
