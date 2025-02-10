from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  

current_dir = os.path.dirname(__file__)
pdf_path = os.path.join(current_dir, "test.pdf")
presistent_directory = os.path.join(current_dir, "db", "chroma_db")

class RagHandler:
    def __init__(self):
        self.current_dir = os.path.dirname(__file__)
        self.presistent_directory = os.path.join(self.current_dir, "db", "chroma_db")

    def create_chroma_storage_from_pdf_directory(self, pdf_directory):
        """
        Loads all PDF files from the provided directory, splits them into chunks,
        adds metadata (e.g., filename) to each document, and creates a persistent
        Chroma vector store if it does not already exist.
        """
        # Only initialize the vector store if it doesn't already exist.
        if not os.path.exists(self.presistent_directory):
            print("Initializing vector store...")
            if not os.path.exists(pdf_directory):
                raise FileNotFoundError(
                    f"The directory {pdf_directory} does not exist. Please check the path."
                )
            
            # Gather all PDF files in the directory.
            pdf_files = [
                os.path.join(pdf_directory, file)
                for file in os.listdir(pdf_directory)
                if file.lower().endswith(".pdf")
            ]
            
            if not pdf_files:
                raise FileNotFoundError("No PDF files found in the specified directory.")
            
            all_docs = []
            for pdf_file in pdf_files:
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                
                # Optionally add metadata (like the filename) to each document.
                for doc in docs:
                    doc.metadata = doc.metadata or {}
                    doc.metadata["source"] = os.path.basename(pdf_file)
                
                all_docs.extend(docs)
            
            # Split the documents into manageable chunks.
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            pdf_chunks = text_splitter.split_documents(all_docs)
            
            # Create embeddings.
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Create and persist the vector store.
            Chroma.from_documents(
                pdf_chunks, embeddings, persist_directory=self.presistent_directory
            )
        else:
            print("Vector store already exists. Skipping initialization.")

    
    def get_relevant_info_from_chroma(self, query):
        # default to mmr retriever if no type is provided
        retriever = self.create_retriever("mmr")
        relevant_docs = retriever.invoke(query)
        return relevant_docs
    
    def create_retriever(self, retriever_type):
        """
        returns a retriever instance based on the provided retriever_type.
        accepted values are:
          - "mmr"
          - "similarity"
          - "similarity_score_threshold"
        """
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=self.presistent_directory, embedding_function=embeddings)
        
        if retriever_type == "mmr":
            retriever = db.as_retriever(
                search_type="mmr", 
                search_kwargs={'k': 6, 'lambda_mult': 0.25}
            )
        elif retriever_type == "similarity":
            retriever = db.as_retriever(
                search_type="similarity", 
                search_kwargs={'k': 6}
            )
        elif retriever_type == "similarity_score_threshold":
            retriever = db.as_retriever(
                search_type="similarity_score_threshold", 
                search_kwargs={'k': 6, 'score_threshold': 0.9}
            )
        else:
            raise ValueError("unsupported retriever type. please choose from 'mmr', 'similarity', or 'similarity_score_threshold'.")
        
        return retriever