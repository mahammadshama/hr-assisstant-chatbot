import os
import streamlit as st
import csv
import tempfile
import speech_recognition as sr
from gtts import gTTS

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from streamlit_chat import message









# ---------------- LLM + Embeddings ----------------
llm = ChatOllama(model="aya:8b")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------- Sidebar ----------------
st.sidebar.title("⚙ HR Bot Settings")

if "lang" not in st.session_state:
    st.session_state.lang = "English"
st.session_state.lang = st.sidebar.radio("🌐 Choose language:", ("English", "Arabic"))

if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = False
st.session_state.voice_enabled = st.sidebar.checkbox("🔊 Enable Voice", value=False)

if st.sidebar.button("🗑 Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.last_input = None

st.sidebar.markdown("💡 Tip: Ask about leave policy, salary, benefits, etc.")


# ---------------- Load CSV ----------------
def load_csv_docs(path, selected_lang):
    docs = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row["language"].strip().lower()
            if lang == selected_lang:
                q = row["question"].strip()
                a = row["answer"].strip()
                docs.append(Document(page_content=f"Q: {q}\nA: {a}", metadata={"language": lang}))
    return docs


if st.session_state.lang == "English":
    docs = load_csv_docs("data/hr_faq1.csv", "en")
else:
    docs = load_csv_docs("data/hr_faq1.csv", "ar")

# ---------------- Vectorstore ----------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})

# ---------------- Prompt ----------------
if st.session_state.lang == "English":
    system_prompt = """You are an HR assistant bot. 
Answer employee questions using the provided context in English only. 
Keep your answers concise (max 3 sentences). 
If the question isn't covered in the context, respond with "Sorry, I don't have information about that. Please contact HR directly". 
Handle greetings and thanks politely.
Context: {context}"""
else:
    system_prompt = """أنت بوت مساعد للموارد البشرية.
أجب عن أسئلة الموظفين باستخدام السياق المقدم فقط باللغة العربية.
يجب أن تكون إجاباتك مختصرة (بحد أقصى 3 جمل).
إذا لم يكن السؤال مغطى في السياق، أجب: "عذراً، ليس لدي معلومات حول ذلك. يرجى الاتصال بالموارد البشرية مباشرة". 
تعامل مع التحية والشكر بلطف.
السياق: {context}"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# ---------------- RAG + History ----------------
history = StreamlitChatMessageHistory()
qa_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, qa_chain)

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history"
)


# ---------------- Voice Helpers ----------------
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.toast("🎙 Listening...")
        audio = r.listen(source, timeout=5, phrase_time_limit=8)
    try:
        return r.recognize_google(audio, language="ar" if st.session_state.lang == "Arabic" else "en")
    except sr.UnknownValueError:
        st.toast("❌ Couldn't understand. Please try again.", icon="⚠")
        return None
    except Exception as e:
        st.toast(f"⚠ Error: {e}")
        return None


def speak_and_save(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name


# ---------------- Page Config + CSS ----------------
st.set_page_config(page_title="HR Assistant Bot", page_icon="💼", layout="wide")

st.markdown("""
<style>
.chat-box {
    height: 70vh;
    overflow-y: auto;
    padding-right: 10px;
}
.user-bubble {
    background-color: #DCF8C6;
    padding: 10px; border-radius: 10px;
    margin: 5px; max-width: 70%;
    float: right; clear: both;
}
.bot-bubble {
    background-color: #FFFFFF;
    padding: 10px; border-radius: 10px;
    margin: 5px; max-width: 70%;
    float: left; clear: both;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

st.title("💼 HR Assistant Chatbot")

# ---------------- Initialize Chat ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_input" not in st.session_state:
    st.session_state.last_input = None

# ---------------- Chat Display ----------------
with st.container():
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(f"<div class='user-bubble'>🧑 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'>🤖 {msg['content']}</div>", unsafe_allow_html=True)
            if msg.get("voice"):
                st.audio(msg["voice"], format="audio/mp3")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Input Section ----------------
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.chat_input("Type your HR question here...")

with col2:
    if st.session_state.voice_enabled and st.button("🎤 Speak"):
        spoken_text = listen()
        if spoken_text:
            user_input = spoken_text

# ---------------- Process Input ----------------
if user_input and user_input != st.session_state.last_input:
    st.session_state.last_input = user_input
    st.session_state.chat_history.append({"role": "user", "content": user_input, "voice": None})

    with st.spinner("💭 Thinking..."):
        try:
            response = chain_with_history.invoke(
                {"input": user_input},
                {"configurable": {"session_id": "abc123"}}
            )
            bot_reply = response['answer']
        except Exception as e:
            bot_reply = f"⚠ Error: {e}"

    # Generate voice if enabled
    audio_file = None
    if st.session_state.voice_enabled:
        lang_code = "ar" if st.session_state.lang == "Arabic" else "en"
        audio_file = speak_and_save(bot_reply, lang=lang_code)

    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply, "voice": audio_file})
    st.rerun()