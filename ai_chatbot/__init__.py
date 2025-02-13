import os
from dotenv import load_dotenv
import gradio as gr
from langchain_groq import ChatGroq
from RagHandler import RagHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import shutil

def chat(user_message, history):
    # Convert the history from dict format into LangChain message objects
    converted_history = []
    previous_jokes = []
    if history:
        for msg in history:
            if msg["role"] == "user":
                converted_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                converted_history.append(SystemMessage(content=msg["content"]))
                # Extract joke if present (assuming it follows the 'Joke:' prefix)
                if "Joke:" in msg["content"]:
                    # Split on "Joke:" and take the later part as the joke text
                    joke_part = msg["content"].split("Joke:")[-1].strip()
                    if joke_part:
                        previous_jokes.append(joke_part)

    # Build a dynamic joke prompt (avoid repeating jokes already told)
    prev_jokes_str = ", ".join(previous_jokes) if previous_jokes else "none"

    dynamic_joke_prompt = (
        f"Generate a very short joke related to the following user input: '{user_message}'. "
        f"The joke should be simple and fun, and assume the user tolerates all kinds of jokes. "
        f"Don't repeat these jokes: {prev_jokes_str}"
    )


    dynamic_joke_message = [SystemMessage(content=dynamic_joke_prompt)]

  
    response = rag_chain.invoke({"input": user_message, "chat_history": converted_history})
    answer_text = response.get("answer", "I'm sorry, I couldn't generate an answer.")

    source_files = []
    if "context" in response:
        for doc in response["context"]:
            source_files.append(doc.metadata["source"])

    source_files = set(source_files)
    # Generate a joke using the dynamic joke prompt
    joke_response = llm.invoke(dynamic_joke_message)
    
    # Get the answer to the user question
    source_text = f"\n\n**Source:** {', '.join(source_files)}" if source_files else ""

    combined_response = f"{answer_text}{source_text}\n\nJoke: {joke_response.content}"
    
    return combined_response


def process_file(uploaded_files):
    pdf_dir = os.path.join(os.path.dirname(__file__), "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        file_name = os.path.basename(uploaded_file.name)
        destination = os.path.join(pdf_dir, file_name)
        shutil.copy(uploaded_file.name, destination)

    ragHandler = RagHandler()
    pdf_directory = os.path.join(os.path.dirname(__file__), "pdfs")
    ragHandler.create_chroma_storage_from_pdf_directory(pdf_directory)

    return "Files uploaded, you can ask questions"

with gr.Blocks() as demo:
    
    file_upload = gr.Interface(
        fn=process_file,
        inputs=gr.File(label="Upload a PDF file",file_types=[".pdf"],file_count='multiple'),
        outputs= gr.Textbox(value="Submit PDF's to ask about them", label="Upload Status")   
    )
    chat_interface = gr.ChatInterface(
        fn=chat,
        title="PDF Chatbot with Jokes",
        description="Ask questions about your PDF",
        type="messages",
        save_history=True
    )


if __name__ == "__main__":
    load_dotenv()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    ragHandler = RagHandler()


    # Create a prompt to reformulate user questions (so that they are standalone)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
        "Also take into account, that you should answer only one question "
        "which user refers to. Don't accumulate answers from multiple questions."
        "Make sure that a rephrased question is a standalone question and doesn't answer previous questions."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ('human', '{input}')
    ])

    # Create a history-aware retriever
    retriever = ragHandler.create_retriever("similarity_score_threshold")
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


    # Create a prompt for answering questions using retrieved context
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. If the knowledge is in the context start with from the pdf:...."
        "If you find the data in the context dont add your knowledge unless asked for"
        "You should also reference the file from which the data was extracted"
        "If the knowledge is not in the context, you can answer using your knowledge, but"
        "You have to specify that the information is not in the pdf and is your knowledge"
        "Answer in the language of the question, if you dont the language ask the user"
        "To change the language of the answer preferably to english"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ('human', '{input}')
    ])

    # Create the chain that uses retrieved documents to answer questions
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Combine the history-aware retriever with the question-answer chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    try:
        demo.launch()
    finally:
        # Cleanup the pdfs directory after the app closes
        pdf_dir = os.path.join(os.path.dirname(__file__), "pdfs")
        if os.path.exists(pdf_dir):
            shutil.rmtree(pdf_dir)