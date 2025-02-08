# ai-chatbot

An AI-powered chatbot that uses LangChain and PDF processing to answer questions from documents.

## Prerequisites

- Python 3.12 (or later, but below 4.0)
- [Poetry](https://python-poetry.org/) installed globally

## Setup

1. **Clone the Repository:**

   ```shell
   git clone https://github.com/OGrocketa/ai-chatot
   cd ai-chatbot

2. **Install dependencies:**
    poetry install


3. **Configure environment variables:**
    Create a .env file in the root directory (if not already present) with the following variables:
    GROQ_API_KEY="your_groq_api_key"
    PDF_ENDPOINT="your_pdf_endpoint_api_key" - not necessary if scraper not used

    If you want to use LangSmith also add:
    LANGSMITH_TRACING=true
    LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
    LANGSMITH_PROJECT="ai-chatbot"
    LANGCHAIN_API_KEY="your_langchain_api_key"

4. **To run the main chatbot application:**
    poetry run python ai_chatbot