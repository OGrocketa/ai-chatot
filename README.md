# ai-chatbot

An AI-powered chatbot with Gradio UI that uses LangChain and PDF processing to answer questions from documents.<br />
AI's used: 
- llama-3.3-70b-versatile (Groq API)<br />
- all-MiniLM-L6-v2 (HuggingFace embeddings)
 
## Prerequisites

- Python 3.12 (or later, but below 4.0)
- [Poetry](https://python-poetry.org/) installed globally

## Setup

1. **Clone the Repository:**

   ```shell
   git clone https://github.com/OGrocketa/ai-chatot
   cd ai-chatbot

2. **Install dependencies:**
    ```shell
    poetry install


3. **Configure environment variables:**
    Create a .env file in the root directory (if not already present) with the following variables:
    GROQ_API_KEY="your_groq_api_key"

4. **To run the main chatbot application:**
    ```shell
    poetry run python ai_chatbot/main.py