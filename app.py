import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    # Debugging: Check the number of indexed documents
    if vector_store.index.ntotal == 0:
        st.error("No documents were indexed in the vector store.")
    else:
        st.success(f"Successfully indexed {vector_store.index.ntotal} documents.")
    
    # Save the vector store locally
    vector_store.save_local("faiss_index")

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the FAISS index to access the document chunks
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search to find the most relevant documents
    docs = new_db.similarity_search(user_question)

    if docs:
        # If relevant documents are found, use them to generate an answer
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])
    else:
        # If no documents are found, generate a general answer using Gemini based on the topic
        st.write("No direct answer found in the documents. Generating a general answer based on the topic...")

        # Generate a response from Gemini based on the overall context (for example, AI models in healthcare)
        response = generate_general_answer(user_question)
        st.write("Reply: ", response)

    # After generating the answer, suggest related questions based on context
    st.write("Related Questions:")
    suggest_related_questions(user_question, new_db)
def generate_general_answer(user_question):
    """
    Generate a general answer based on the topic of the document if no direct answer is found.
    This uses the model (Gemini) to generate a meaningful response.
    """
    # For example, if the document discusses deep learning or healthcare, generate an answer based on that.
    # Here, we use a general context related to healthcare, AI, or deep learning.
    
    # Constructing a relevant fallback context or using Gemini's capabilities for topic-based generation
    prompt_template = """
    Answer the following question based on general knowledge of deep learning and healthcare:
    Question: {question}

    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    chain = LLMChain(prompt=prompt, llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3))

    # Generate a response based on the fallback context
    response = chain.run({"question": user_question})
    return response
def suggest_related_questions(user_question, new_db):
    """
    Suggest related questions strictly based on the content retrieved from the document chunks.
    """
    # Perform similarity search to find the most relevant document chunks
    related_docs = new_db.similarity_search(user_question)

    if related_docs:
        # Extract the context from the document chunks
        context = " ".join([doc.page_content for doc in related_docs])

        # Ensure context is not empty
        if context.strip():
            # Prompt template for generating related questions
            prompt_template = """
            Based on the following document context, suggest 3 questions that can be answered using this information.
            Ensure the questions are strictly based on the provided context:
            Context: {context}
            Suggested questions:
            """

            # Create a prompt using the extracted context
            prompt = PromptTemplate(template=prompt_template, input_variables=["context"])

            # Use LLMChain with Gemini to generate the related questions
            chain = LLMChain(prompt=prompt, llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3))

            # Generate related questions
            related_questions = chain.run({"context": context})

            # Display the related questions
            st.write("Related Questions:")
            st.write(related_questions)
        else:
            st.write("No sufficient context in the document to generate related questions.")
    else:
        st.write("No relevant content found in the document to suggest related questions.")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def get_context_from_db(new_db):
    """
    Get context (a snippet) from the database if no direct match is found.
    This will be used by the Gemini model to generate an answer.
    """
    # Select top 3 relevant documents to form context for the Gemini model
    docs = new_db.similarity_search("")  # Empty search just to get top documents based on the entire chunk data
    context = " ".join([doc.page_content[:500] for doc in docs[:3]])  # Use the first 3 documents
    return context

def main():
    st.set_page_config("Chat PDF",page_icon="::books")
    st.header("Chat with your PDFs:books:")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
