💼 HR Assistant Chatbot

An intelligent AI-powered HR Assistant Bot built using LangChain, Ollama, Streamlit, and Hugging Face embeddings, capable of understanding both English and Arabic, with voice interaction and context-aware answers.

🌟 Overview

The HR Assistant Chatbot is designed to automate HR-related employee queries in an organization. It combines retrieval-augmented generation (RAG) with LLM intelligence to answer questions using data from internal HR policies.

It also supports voice input and audio replies, providing a more natural, human-like interaction experience.

This project demonstrates end-to-end integration of:

LangChain RAG pipelines

Ollama local LLM inference

Multi-language support (English & Arabic)

Speech Recognition and Text-to-Speech (TTS)

Streamlit web interface with WhatsApp-style chat UI

🧠 Key Features

✅ HR Query Understanding:
Automatically answers employee questions related to leave policies, salary, benefits, and other HR topics.

✅ Context-Aware Retrieval (RAG):
Uses company HR FAQs stored in a CSV to fetch the most relevant response using Chroma VectorDB and Hugging Face embeddings.

✅ Multilingual Support:
Fully supports English and Arabic, including text, voice input, and voice output.

✅ Voice Interaction:

🎙 Speak to the bot instead of typing.

🔊 The bot replies back in spoken audio using gTTS.

✅ Smart Conversation Memory:
Maintains chat context (previous messages) using LangChain’s memory system, making interactions natural and continuous.

✅ Modern UI:

Built with Streamlit

Polished WhatsApp-style design for easy readability

Real-time chat bubbles for user and bot

✅ Offline LLM:
Uses Ollama with Aya 8B model, so responses are generated locally — no dependency on paid APIs like OpenAI.

⚙️ Tech Stack                    
Category	                    Tools & Libraries                 
Frontend / UI	                Streamlit, Streamlit-Chat               
LLM / NLP	                    Ollama (Aya-8B model)                
Embeddings	                  Hugging Face – sentence-transformers/all-MiniLM-L6-v2                
Vector Store	                ChromaDB                  
Framework	                    LangChain          
Speech Recognition	          SpeechRecognition                     
Text-to-Speech	              gTTS                  
Language Support	            English, Arabic               
IDE	                          PyCharm
Version Control	              Git + GitHub

🗂️ Project Structure                   
hr-assistant-chatbot/                         
│                    
├── data/                                      
│   └── hr_faq1.csv                # HR policy FAQs (English + Arabic)                    
│
├── main.py                        # Main Streamlit app file                  
│
├── requirements.txt               # Python dependencies                          
│
├── README.md                      # Project documentation (this file)                  
│
└── .gitignore                     # To exclude unnecessary files              

⚡ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/yourusername/hr-assistant-chatbot.git
cd hr-assistant-chatbot

2️⃣ Create a virtual environment
python -m venv .venv


Activate it:

Windows: .\.venv\Scripts\activate

Linux/Mac: source .venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run Ollama (make sure it’s installed)
ollama run aya:8b

5️⃣ Launch the app
streamlit run main.py


Then open the local URL (usually http://localhost:8501
).

🧩 How It Works

CSV Data Load:
HR FAQs (with English and Arabic questions/answers) are loaded into memory and split into chunks.

Vectorization:
Using Hugging Face embeddings, the documents are converted into dense vector representations stored in ChromaDB.

Retrieval:
When a user asks a question, the chatbot retrieves the most similar HR FAQ chunk using MMR similarity search.

LLM Response:
The Aya 8B model (Ollama) processes the retrieved text and formulates a human-like answer.

Memory Tracking:
Conversation context is preserved using LangChain message history, allowing follow-up questions.

Voice Interaction (optional):

Speech input captured using SpeechRecognition.

Bot’s spoken reply generated using gTTS.

🗣 Example Interaction

User (English):

Hi, what is the annual leave policy?

Bot (English):

Employees are entitled to 18 days of paid annual leave.

User (Arabic):

ما هي سياسة الإجازة السنوية؟

Bot (Arabic):

يحق للموظفين الحصول على 18 يومًا من الإجازة السنوية المدفوعة الأجر.

📦 requirements.txt

Here’s the exact dependencies for your project:

streamlit
streamlit-chat
langchain
langchain-community
langchain-ollama
langchain-core
chromadb
sentence-transformers
torch
gtts
SpeechRecognition
pydub


(Optional for GPU users:)

accelerate
bitsandbytes

📸 Screenshots
💬 English Chat Mode

(add your image here)

🗣 Arabic Voice Mode

(add your image here)

🚀 Future Improvements

Add document upload (PDF/Docx HR files) support.

Integrate OpenAI or Gemini for cloud-based inference.

Save chat history for each user.

Add authentication and role-based access (e.g., admin HR panel).

👩‍💻 Author

Mahammad Sha
📍 Data Scientist | Machine Learning Enthusiast
💡 Passionate about AI + NLP + LLM Applications
🌐 GitHub

⭐ Contribute

Want to improve this project? Fork it, create a branch, and submit a pull request — contributions are always welcome!

📄 License

This project is licensed under the MIT License — free to use and modify with attribution.
