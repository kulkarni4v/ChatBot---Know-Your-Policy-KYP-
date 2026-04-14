# ChatBot---Know-Your-Policy-KYP-

## Overview

Know Your Policy (KYP) is an intelligent AI chatbot designed to help you understand complex documents like insurance policies, service agreements, and legal contracts. It uses a Retrieval-Augmented Generation (RAG) architecture to provide answers that are strictly based on the content of your uploaded document, preventing AI hallucinations and ensuring factual accuracy.

This application features an interactive, modern web interface built with Streamlit, allowing you to upload any PDF and start asking questions immediately.

## Features

*   **Strictly Document-Based Answers:** The RAG pipeline ensures the chatbot answers questions **only** from the provided document context, eliminating guesswork and external knowledge.
*   **Interactive Chat UI:** A user-friendly, responsive interface powered by Streamlit, featuring ChatGPT-like streaming responses.
*   **Upload Any PDF:** Easily upload your own insurance policy, legal agreement, or any other PDF document for analysis.
*   **Conversational Memory:** The chatbot remembers previous questions and answers in the conversation, allowing for contextual follow-up queries.
*   **Local Vector Storage:** Utilizes ChromaDB to create and persist a local vector database of your document, enabling instant loading for future sessions without reprocessing.
*   **Optimized for Cost and Performance:** Built with efficient models like OpenAI's `gpt-4o-mini` and `text-embedding-3-large` for a balance of speed, accuracy, and low cost.

## How It Works

The chatbot follows a standard RAG (Retrieval-Augmented Generation) pipeline to process and query your documents:

1.  **Document Loading & Processing:** When a PDF is provided, it's loaded and split into smaller, manageable text chunks. This process maintains the continuity of the text to preserve context.
2.  **Embedding & Indexing:** Each text chunk is converted into a numerical vector (embedding) using OpenAI's embedding models. These embeddings capture the semantic meaning of the text.
3.  **Vector Storage:** The embeddings and their corresponding text are stored in a local ChromaDB vector store. This creates a searchable index of your document.
4.  **Retrieval:** When you ask a question, the system converts your query into an embedding and uses it to find the most semantically similar text chunks from the ChromaDB index.
5.  **Augmented Generation:** The retrieved text chunks (the "context"), along with the conversation history and your original question, are fed into an LLM (OpenAI's GPT-4o-mini). A carefully crafted prompt instructs the model to generate an answer **strictly based on the provided context**.
6.  **Streaming Response:** The generated answer is streamed back to the user interface, providing a real-time, interactive experience.

## Tech Stack

*   **Core Framework:** [LangChain](https://www.langchain.com/)
*   **User Interface:** [Streamlit](https://streamlit.io/)
*   **LLM:** OpenAI GPT-4o-mini
*   **Embeddings:** OpenAI text-embedding-3-large
*   **Vector Database:** [ChromaDB](https://www.trychroma.com/)
*   **PDF Processing:** PyPDF

## Setup and Usage

Follow these steps to run the KYP Chatbot on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/kulkarni4v/Langchain-Based-AI-ChatBot---Know-Your-Policy-KYP-.git
cd Langchain-Based-AI-ChatBot---Know-Your-Policy-KYP-
```

### 2. Create a Virtual Environment and Install Dependencies

It's recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### 3. Configure Your OpenAI API Key

The application uses Streamlit's secrets management to handle the API key securely.

1.  Create a directory named `.streamlit` in the root of the project directory if it doesn't exist.
2.  Inside `.streamlit`, create a file named `secrets.toml`.
3.  Add your OpenAI API key to this file as follows:

    ```toml
    # .streamlit/secrets.toml

    OPENAI_API_KEY = "your-openai-api-key-here"
    ```

### 4. Run the Streamlit Application

Launch the web application with the following command:

```bash
streamlit run streamlit_app.py
```

Your web browser will open a new tab with the chatbot interface.

### 5. Using the Chatbot

1.  **Select a Document:** Use the sidebar to either **Upload a PDF** from your computer or use the default **File Path**.
2.  **Initialize the Chatbot:** Click the **"Initialize Chatbot"** button. This will process the PDF, create embeddings, and set up the RAG chain. For subsequent uses with the same document, the chatbot will load the existing database from the `chroma_db` directory.
3.  **Start Asking Questions:** Once the chatbot is ready, you can start typing your questions into the chat input at the bottom of the screen.
