# NeuroPDF: AI-Powered Document Chat 🤖

**NeuroPDF** is a Streamlit-based application that allows users to chat with a conversational AI model trained on PDF documents. The chatbot extracts information from uploaded PDF files and answers user questions based on the provided context.

🔗 Live App: [https://gmultichat.streamlit.app/](https://gmultichat.streamlit.app/)

🎥 Demo:  
![Demo](https://github.com/kaifcoder/gemini_multipdf_chat/assets/57701861/f6a841af-a92d-4e54-a4fd-4a52117e17f6)

---

## 🚀 Features

- **📁 PDF Upload** – Upload and process multiple PDFs at once.
- **📝 Text Extraction** – Automatically extracts and chunks PDF text.
- **🧠 Conversational AI** – Powered by Google Gemini AI for accurate answers.
- **💬 Interactive Chat** – Intuitive chat interface using Streamlit.

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.10+
- Google API Key (for Gemini model)

### Installation

1. **Clone the Repository**

```bash
git clone https://github.com/KarthikB6360/NeuroPDF-
cd neuropdf-chatbot

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google API Key:**
   - Obtain a Google API key and set it in the `.env` file.

   ```bash
   GOOGLE_API_KEY=your_api_key_here
   ```

4. **Run the Application:**

   ```bash
   streamlit run main.py
   ```

5. **Upload PDFs:**
   - Use the sidebar to upload PDF files.
   - Click on "Submit & Process" to extract text and generate embeddings.

6. **Chat Interface:**
   - Chat with the AI in the main interface.

## Project Structure

- `app.py`: Main application script.
- `.env`: file which will contain your environment variable.
- `requirements.txt`: Python packages required for working of the app.
- `README.md`: Project documentation.

## Dependencies

- PyPDF2
- langchain
- Streamlit
- google.generativeai
- dotenv

## Acknowledgments

- [Google Gemini](https://ai.google.com/): For providing the underlying language model.
- [Streamlit](https://streamlit.io/): For the user interface framework.
