import fitz  # PyMuPDF
import langchain as lc
import openai
import os
import streamlit as st

# Access the OpenAI API key from Replit secret
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_file):
    """
    Extract text from the provided PDF file.

    :param pdf_file: Uploaded PDF file.
    :return: Extracted text as a string.
    """
    text_content = ''
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text_content += page.get_text()
    return text_content

def main():
    st.title("Enhanced Knowledge AI Chatbot")
    uploaded_file = st.file_uploader("Upload a PDF file to enhance the chatbot's knowledge:", type="pdf")

    if uploaded_file is not None:
        knowledge_text = extract_text_from_pdf(uploaded_file)
        user_input = st.text_input("Ask a question:")

        if user_input:
            openai_api = lc.OpenAICompletionAPI(engine="text-davinci-003")
            chain = lc.LLMChain(llm_api=openai_api)
            context = lc.ContextManager(contexts=[knowledge_text])
            chat_response = chain.run(prompt=user_input, context_manager=context)
            st.text_area("Response", value=chat_response, height=150, max_chars=10000)

if __name__ == "__main__":
    main()
