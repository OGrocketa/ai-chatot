import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from RagHandler import RagHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

ragHandler = RagHandler()
pdf_directory = os.path.join(os.path.dirname(__file__), "../pdfs")
ragHandler.create_chroma_storage_from_pdf_directory(pdf_directory)

# Create a system prompt to reformulate user questions by considering previous chat history
# This helps make questions stand-alone for retrieval purposes
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ('human','{input}')
])

retriever = ragHandler.create_retriever("mmr")

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Define the question-answering system prompt which instructs the AI on how to answer using retrieved context
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. If the knowledge is in the context start with from the pdf:...."
    " If the knwoledge is not in the context, you can answer using your knwoledge"
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ('human','{input}')
])

# Create a chain that integrates retrieved documents to directly answer questions
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create the full retrieval chain by combining the history-aware retriever and question-answer chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

joke_message = [SystemMessage(content="Act as a world class stand-up comedian."
                                "Genearte a very short joke about humans.")]



def chatbot():
    print("\nWelcome to the AI chatbot. Ask me anything about your pdf!\nType 'e' to exit.")
    chat_history = []
    joke_history = []
    while True:
        user_input = input("You: ")

        if user_input.lower() == "e":
            print("Goodbye!")
            break
        
        # Generate a dynamic joke prompt that avoids repeating jokes previously told
        previous_jokes = ", ".join(joke_history) if joke_history else "none"
        dynamic_joke_prompt = (
            f"Generate a very short joke about humans any type of joke is fine. "
            f"Don't repeat these jokes: {previous_jokes}."
        )
        dynamic_joke_message = [SystemMessage(content=dynamic_joke_prompt)]
        
        joke_response = llm.invoke(dynamic_joke_message)
        response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})

        
        print("--------------------------------")
        print("AI: ", response["answer"], "\n\nJoke:",joke_response.content)
        print("--------------------------------\n")

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(SystemMessage(content=response["answer"]))
        joke_history.append(joke_response.content)
        
if __name__ == "__main__":
    chatbot()