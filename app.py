import streamlit as st
import nltk
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler

nltk.download('punkt_tab')

class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.response = ""

    def on_llm_start(self, *arg, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *arg, **kwargs):
        save_message(self.response, "ai")

    def on_llm_new_token(self, token, *arg, **kwargs):
        self.response += token
        self.message_box.markdown(self.response)


if "openai_api" not in st.session_state:
    st.session_state["openai_api"] = ""


@st.cache_resource(show_spinner="LLM 연결중(1)...")
def generate_llm(api_key):
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-0125",
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        api_key=api_key,
    )
    return llm

@st.cache_resource(show_spinner="LLM 연결중(2)...")
def generate_memory_llm(api_key):
    memory_llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-0125",
        api_key=api_key,
    )

    st.session_state["memory"] = ConversationBufferMemory(
        llm=memory_llm,
        max_token_limit=150,
        memory_key="history",
        return_messages=True,
    )
    return st.session_state["memory"]

def save_memory(input, output):
    memory.save_context({"input": input}, {"output": output})


def load_memory(_):
    return memory.load_memory_variables({})["history"]


def format_doc(document):
    return "\n\n".join(doc.page_content for doc in document)


def invoke_chain(question):
    result = chain.invoke(question).content
    save_memory(question, result)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], False)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
        if save:
            save_message(message, role)


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_name = file.name
    file_path = f"./{file_name}"
    file_content = file.read()
    with open(file_path, "wb") as f:
        f.write(file_content)
    st.write("Success1")
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=50,
    )
    st.write("Success2")
    docs = loader.load_and_split(text_splitter=splitter)
    st.write("Success3")
    embedder = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedder)
    st.write("Success4")
    return vectorStore.as_retriever()


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant. Answer questions using only the following context. and You remember conversations with human.
            DON'T make it up.
            --------
            {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "memory" not in st.session_state:
    st.session_state["memory"] = ""

st.title("Document GPT")

st.markdown(
    """
    Welcome!
    Use this chatbot to ask questions to an AI about your files!
    Upload your files on the sidebar.
    """
)

with st.sidebar:
    llm_api = st.text_input("Input OpenAI API")
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["txt", "pdf", "docx"],
    )

if file:
    if not llm_api.startswith("sk-"):
        st.warning("Please enter your OpenAI Key!", icon="⚠️")
    else:
        llm = generate_llm(llm_api)
        memory = generate_memory_llm(llm_api)

        retriever = embed_file(file)
        send_message("How can I help you?", "ai", save=False)
        paint_history()
        answer = st.chat_input("Ask anything about your file....")
        if answer:
            send_message(answer, "human", True)
            chain = (
                {
                    "context": retriever | RunnableLambda(format_doc),
                    "history": RunnableLambda(load_memory),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                invoke_chain(answer)
else:
    st.session_state["messages"] = []
