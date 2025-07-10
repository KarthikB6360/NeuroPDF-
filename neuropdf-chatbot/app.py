import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from gtts import gTTS
import tempfile
import fitz  # PyMuPDF
from googletrans import Translator

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
genai.configure(api_key=api_key)

translator = Translator()

# --- Text-to-Speech ---
def speak_text(text):
    try:
        tts = gTTS(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            audio_path = fp.name
        with open(audio_path, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")
    except Exception as e:
        st.warning(f"Audio error: {e}")

# --- Translation ---
def translate_input_output(text, target_lang="en"):
    detected = translator.detect(text)
    translated_input = translator.translate(text, dest="en").text
    return translated_input, detected.lang

def translate_back(text, dest_lang):
    return translator.translate(text, dest=dest_lang).text

# --- PDF Text Extraction ---
def get_pdf_text_and_pages(pdf_docs):
    texts = []
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for i, page in enumerate(reader.pages):
            content = page.extract_text()
            if content:
                texts.append(Document(page_content=content, metadata={"page": i + 1}))
    return texts

# --- Visual Extraction ---
def extract_images_from_pdfs(pdf_docs):
    image_data = []
    for pdf in pdf_docs:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_tag = f"<Image page={page_index+1} image={img_index+1}>"
                image_data.append({
                    "image": image_bytes,
                    "ext": image_ext,
                    "page": page_index + 1,
                    "text_tag": image_tag
                })
    return image_data

# --- Chunking & Embedding ---
def get_text_chunks_with_metadata(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

def get_vector_store(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    vectorstore.save_local("faiss_index")

# --- Conversational Chain ---
def get_conversational_chain(vectorstore):
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
    )

def user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    chain = get_conversational_chain(vectorstore)
    return chain({"question": question, "chat_history": []})

# --- Chat Reset ---
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload PDFs and ask a question in any language."}
    ]

# --- Main App ---
def main():
    st.set_page_config(page_title="PDF Chatbot with KAG + MCP + Visual + Voice", page_icon="📄")

    st.title("📄 Ask Your PDF with Knowledge-Aware + Multi-Context Prompting")

    with st.sidebar:
        st.header("📁 Upload PDFs")
        pdf_docs = st.file_uploader("Choose your PDFs", accept_multiple_files=True)
        if st.button("📌 Process PDFs"):
            with st.spinner("Processing..."):
                docs = get_pdf_text_and_pages(pdf_docs)
                images = extract_images_from_pdfs(pdf_docs)
                chunks = get_text_chunks_with_metadata(docs)
                get_vector_store(chunks)
                st.session_state.image_data = images
                st.success("✅ Done! Start asking questions.")

    st.sidebar.button("🧹 Clear Chat", on_click=clear_chat_history)
    if "messages" not in st.session_state:
        clear_chat_history()
    if "image_data" not in st.session_state:
        st.session_state.image_data = []

    if st.session_state.image_data:
        st.subheader("🖼️ Visuals Extracted:")
        for img in st.session_state.image_data:
            st.image(img["image"], caption=f"{img['text_tag']} on Page {img['page']}", use_column_width=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    show_highlight = st.toggle("🔍 Highlight Context", value=False)
    prompt_style = st.selectbox("🧠 Prompt Style", ["Default", "ELI5", "Formal", "Creative"])

    if prompt := st.chat_input("Ask anything about the PDF (any language)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    translated_prompt, original_lang = translate_input_output(prompt)

                    # --- MCP + KAG Prompt Construction ---
                    prompt_context = "Use facts strictly from the PDF. "
                    if prompt_style == "ELI5":
                        prompt_context = "Explain like I’m 5. " + prompt_context
                    elif prompt_style == "Formal":
                        prompt_context = "Respond formally. " + prompt_context
                    elif prompt_style == "Creative":
                        prompt_context = "Answer in a creative storytelling style. " + prompt_context

                    if st.session_state.image_data:
                        prompt_context += "There are visuals present that may be relevant. "

                    final_prompt = prompt_context + translated_prompt
                    result = user_input(final_prompt)

                    answer = result["answer"]
                    sources = result["source_documents"]
                    translated_answer = translate_back(answer, original_lang)

                    full_response = f"**Answer ({original_lang.upper()}):**\n{translated_answer}"

                    if show_highlight:
                        context_display = ""
                        for src in sources:
                            page_num = src.metadata.get("page", "N/A")
                            content = src.page_content.strip().replace("\n", " ")
                            context_display += f"🔍 **Page {page_num}**\n> {content[:500]}...\n\n"
                        full_response += f"\n\n---\n📘 **Knowledge Used (KAG):**\n{context_display}"

                    st.markdown(full_response)
                    speak_text(translated_answer)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
