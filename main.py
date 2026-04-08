"""
Health Insurance Policy RAG Chatbot
A chatbot that reads PDF insurance policies, stores them in ChromaDB, 
and answers queries based only on the policy content with conversation memory.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ============================================================================
# CONFIGURATION - Load API key from environment variable
# ============================================================================
# API key is loaded from environment variable for security
# This keeps your key out of version control and source code
# from dotenv import load_dotenv

# # load_dotenv(): Reads variables from .env file and loads them into environment
# # - Looks for a .env file in the same directory as this script
# # - Makes variables accessible via os.environ
# load_dotenv()

# # Get the API key from environment variable
# # - Will be None if not set, which we check below
# api_key = os.getenv("OPENAI_API_KEY")

import streamlit as st
api_key=st.secrets["OPENAI_API_KEY"]


# Validate that the API key exists
if not api_key:
    raise ValueError(
        "OpenAI API key not found! Please set it in one of these ways:\n"
        "1. Create a .env file with: OPENAI_API_KEY=your-key-here\n"
        "2.  Please set it in Streamlit secrets as OPENAI_API_KEY."
    )

# ============================================================================
# STEP 1: Load and Process the PDF Policy Document
# ============================================================================

def load_and_process_pdf(pdf_path):
    """
    Load a PDF file and split it into manageable chunks for processing.

    Args:
        pdf_path: Path to the PDF insurance policy file

    Returns:
        List of document chunks ready for embedding
    """
    # PyPDFLoader: Loads PDF and extracts text page by page
    # Each page becomes a separate document with metadata (page number)
    loader = PyPDFLoader(pdf_path)

    # load(): Reads the entire PDF and returns a list of Document objects
    documents = loader.load()

    print(f"✓ Loaded {len(documents)} pages from PDF")

    # RecursiveCharacterTextSplitter: Intelligently splits text into chunks
    # - chunk_size=1000: Each chunk will be ~1000 characters (balances context vs cost)
    # - chunk_overlap=200: Overlaps chunks by 200 chars to maintain context continuity
    # - Tries to split on paragraph breaks, then sentences, then words to keep meaning intact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Priority order for splitting
    )

    # split_documents(): Breaks documents into smaller chunks
    chunks = text_splitter.split_documents(documents)

    print(f"✓ Split into {len(chunks)} chunks")

    return chunks

# ============================================================================
# STEP 2: Create Embeddings and Store in ChromaDB
# ============================================================================

def create_vector_store(chunks, persist_directory="./chroma_db"):
    """
    Convert text chunks to embeddings and store in ChromaDB for semantic search.

    Args:
        chunks: List of document chunks from Step 1
        persist_directory: Local folder to save the database

    Returns:
        ChromaDB vector store instance
    """
    # OpenAIEmbeddings with text-embedding-3-small: The cheapest and smallest OpenAI embedding model
    # - Converts text into 1536-dimensional vectors that capture semantic meaning
    # - Cost: $0.02 per 1M tokens (very cheap!)
    # - Faster than older models like text-embedding-ada-002
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    print("✓ Creating embeddings and storing in ChromaDB...")

    # Chroma.from_documents(): Creates a vector database from document chunks
    # - Generates embeddings for each chunk using OpenAI API
    # - Stores embeddings + original text + metadata in ChromaDB
    # - persist_directory: Saves DB to disk so you don't re-process the PDF every time
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"✓ Vector store created and persisted to {persist_directory}")

    return vectorstore

# ============================================================================
# STEP 3: Load Existing Vector Store (Skip re-processing if DB exists)
# ============================================================================

def load_vector_store(persist_directory="./chroma_db"):
    """
    Load an existing ChromaDB vector store from disk.

    Args:
        persist_directory: Folder where the database was saved

    Returns:
        Loaded ChromaDB vector store
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=api_key)

    # Chroma(): Loads existing database from disk without re-creating embeddings
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    print(f"✓ Loaded existing vector store from {persist_directory}")

    return vectorstore

# ============================================================================
# STEP 4: Create the RAG Chatbot with Memory
# ============================================================================

def create_rag_chain(vectorstore):
    """
    Build a conversational RAG chain with memory that answers based on policy only.

    Args:
        vectorstore: ChromaDB instance with policy embeddings

    Returns:
        Conversational chain ready for Q&A
    """
    # ChatOpenAI with gpt-4o-mini: The cheapest GPT model available
    # - Cost: $0.150 per 1M input tokens, $0.600 per 1M output tokens
    # - Much cheaper than gpt-4 or gpt-3.5-turbo
    # - temperature=0: Makes responses deterministic and factual (no creativity)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key  # 0 = factual, 1 = creative
    )

    # ConversationBufferMemory: Stores the entire chat history
    # - memory_key="chat_history": Key name for accessing history in prompts
    # - return_messages=True: Returns messages as Message objects (better for chains)
    # - output_key="answer": Specifies which output to store in memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    system_template = """
        You are an expert in reading and interpreting policies, agreements, and legal documents.
        These documents may contain complex wording, implicit meanings, and technical clauses.
        Your job is to:
        - Carefully analyze the provided policy document
        - Answer strictly based on the document content (explicit or implicit meaning)
        STRICT RULES:
        - Do NOT go beyond the document
        - Do NOT assume or imagine anything
        - Do NOT add external knowledge
        - Always stick to wording from the document
        - Important terms (like "Reasonable and Customary Charges") must not be missed
        ANSWERING STYLE:
        - No preamble (no greetings or introductions)
        - Be precise and to the point
        - Use clear paragraphs
        - Leave a blank line between sections
        - Use bullet points where appropriate
        - Include numbers, rules, and figures if present in the document
        
        INPUT VALIDATION:
        - If the question is not meaningful or not related to the policy or having a bad intent or disrespectful or related to adultry, terrorism ,religion, any celebrity or nation or community
        or any illicit material or intent.
        Then Politely respond:
        Please restrict your queries to the uploaded policy document. 
        Do not answer such queries even if they are reframed or re-asked implicitly.
        Please give a disclaimer before responfing to first time in chat, that >> Iam not intelligent as human, im AI , and AI can make mistakes, 
        and I will not be accountable for your thoughts/actions influenced by this conversation"""
    human_template = """Context from policy:{context}
                Chat History:
                {chat_history}
                Question:
                {question}
                Answer (based only on the policy):
                """


    from langchain.prompts import ChatPromptTemplate
    
    PROMPT = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])

    # # PromptTemplate: Structures the input to the LLM with our custom instructions
    # PROMPT = PromptTemplate(
    #     template=prompt_template,
    #     input_variables=["context", "chat_history", "question"]
    # )
    
    # ConversationalRetrievalChain: Combines retrieval + chat + memory
    # - retriever: Searches vectorstore for relevant chunks (k=3 returns top 3 matches)
    # - memory: Maintains conversation history for context
    # - combine_docs_chain_kwargs: Passes our custom prompt
    # - return_source_documents=True: Returns which chunks were used (for transparency)
    # - verbose=False: Set to True to see internal chain steps
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        ),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=False
    )

    print("✓ RAG chain created with conversation memory")

    return chain

# ============================================================================
# STEP 5: Interactive Chatbot Interface
# ============================================================================

def chat_interface(chain):
    """
    Simple command-line chat interface for interacting with the policy chatbot.

    Args:
        chain: The conversational RAG chain
    """
    print("\n" + "="*70)
    print("🏥 Health Insurance Policy Chatbot")
    print("="*70)
    print("Ask questions about your health insurance policy.")
    print("Type 'quit', 'exit', or 'q' to end the conversation.")
    print("="*70 + "\n")

    while True:
        # Get user input
        user_question = input("You: ").strip()

        # Exit conditions
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using the Health Insurance Policy Chatbot!")
            break

        # Skip empty inputs
        if not user_question:
            continue

        # Process the question through the RAG chain
        # chain(): Takes a question and returns answer + source documents
        result = chain({"question": user_question})

        # Extract the answer from the result
        answer = result["answer"]

        # Print the chatbot's response
        print(f"\nChatbot: {answer}\n")

        # Optional: Show source documents for transparency
        # Uncomment below to see which policy sections were used
        # print("\n📄 Sources used:")
        # for i, doc in enumerate(result["source_documents"], 1):
        #     print(f"{i}. Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:100]}...")

# ============================================================================
# MAIN EXECUTION FLOW
# ============================================================================

def main():
    """
    Main workflow: Load PDF → Create/Load DB → Start Chatbot
    """
    # Path to your health insurance policy PDF
    pdf_path = "data_pdf_file/care-supreme---policy-terms-&-conditions-(effective-from-19-march-2025).pdf"
    db_path = "./chroma_db"

    # Check if we need to process the PDF or can load existing DB
    if not os.path.exists(db_path):
        print("📄 Processing PDF for the first time...")

        # Step 1: Load and chunk the PDF
        chunks = load_and_process_pdf(pdf_path)

        # Step 2: Create embeddings and store in ChromaDB
        vectorstore = create_vector_store(chunks, db_path)
    else:
        print("📂 Loading existing vector database...")

        # Step 3: Load existing database
        vectorstore = load_vector_store(db_path)

    # Step 4: Create the RAG chain with memory
    chain = create_rag_chain(vectorstore)

    # Step 5: Start the chat interface
    chat_interface(chain)

# ============================================================================
# RUN THE CHATBOT
# ============================================================================

if __name__ == "__main__":
    main()
