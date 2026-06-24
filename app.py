import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# Page setup
st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🤖", layout="centered")
st.title("🤖 Zyro Dynamics HR Help Desk")
st.write("Ask any questions regarding company HR policies, leaves, or conduct.")

# Check for API Key
if "GOOGLE_API_KEY" not in os.environ:
    # Streamlit Secrets nunchi load avthundhi clear ga
    if st.secrets and "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        st.error("❌ GOOGLE_API_KEY not found in Advanced Settings Secrets!")
        st.stop()

# Cache initialization so loading happens only once
@st.cache_resource
def initialize_rag_pipeline():
    corpus_path = "./hr_docs"
    
    # 1. Load Documents
    if not os.path.exists(corpus_path) or not os.listdir(corpus_path):
        return None, "Error: './hr_docs' folder is empty or missing in repository!"
        
    loader = PyPDFDirectoryLoader(corpus_path)
    documents = loader.load()
    
    # 2. Chunk Documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    # 3. Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 4. Initialize LLM (Gemini 2.5 Flash used)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    # 5. Prompts Setup
    rag_prompt = ChatPromptTemplate.from_template("""
    You are an HR assistant for Zyro Dynamics.
    Use ONLY the provided context to answer the employee's question.
    Rules:
    1. Answer only from the context.
    2. If the answer is not available in the context, say:
       "I could not find this information in the HR policy documents."
    3. Be concise and professional.
    Context:
    {context}
    Question: {question}
    Answer:""")
    
    oos_prompt = ChatPromptTemplate.from_template("""
    You are an AI guardrail for an HR Help Desk. 
    Analyze the user's question. If it is NOT related to corporate HR policies, company profile, leave, work from home, code of conduct, travel expenses, performance reviews, or IT policies, reply with 'True'. Otherwise, reply with 'False'.
    Question: {question}
    Answer:""")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    guard_chain = oos_prompt | llm | StrOutputParser()
    
    return {"rag": rag_chain, "guard": guard_chain}, "Success"

# Run pipeline setup
pipeline, status_msg = initialize_rag_pipeline()

if pipeline is None:
    st.error(status_msg)
    st.stop()

# Chat History setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chats
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt logic
if user_query := st.chat_input("Ask your HR query here..."):
    # Display user chat
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Guardrail evaluation step
    with st.spinner("Thinking..."):
        is_oos = pipeline["guard"].invoke({"question": user_query}).strip().lower()
        
        if "true" in is_oos:
            response = "I am an HR assistant and can only answer questions related to Zyro Dynamics HR policies."
        else:
            response = pipeline["rag"].invoke(user_query)
            
    # Display bot answer
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
