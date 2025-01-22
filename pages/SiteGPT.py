import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.vectorstores import FAISS
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough


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


def generate_llm(openAI_KEY):
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-0125",
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        api_key=openAI_KEY,
    )
    return llm

@st.cache_resource(show_spinner="Search Site Information...")
def get_retriever(url, openAI_KEY):
    try:
        if not os.path.exists("./.cache/embeddings"):
            os.makedirs("./.cache/embeddings")
        url_name = (
            str(url).replace("https://", "").replace(".", "").replace("/sitemapxml", "")
        )
        cache_dir = LocalFileStore(f"./.cache/embeddings/{url_name}")
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
        )
        loader = SitemapLoader(
            url,
            # filter_urls=[],
            filter_urls=(
                [
                    r"https:\/\/developers.cloudflare.com/ai-gateway.*",
                    r"https:\/\/developers.cloudflare.com/vectorize.*",
                    r"https:\/\/developers.cloudflare.com/workers-ai.*",
                ]
            ),
        )
        loader.requests_per_second = 5
        docs = loader.load_and_split(splitter)
        embedder = OpenAIEmbeddings(api_key=openAI_KEY)
        cache_embedder = CacheBackedEmbeddings.from_bytes_store(embedder, cache_dir)
        vectorStore = FAISS.from_documents(docs, cache_embedder)

        return vectorStore.as_retriever()
    except Exception as e:
        st.error("Failed to Load Site Information")
        return "Error"


def paint_messages():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], False)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
        if save:
            save_message(message, role)


class Prompts:
    def get_answers_prompt(self):
        return ChatPromptTemplate.from_template(
            """
        Using ONLY the following context answer the user's question. If you can't answer,
        Just say you don't know, don't make anyting up.

        Then, give a score to the answer between 0 and 5. 0 being not helpful to
        the user and 5 being helpful to the user.

        Make sure to include the answer's score.
        ONLY one result should be output.

        Context : {context}

        Examples:

        Question: How far away is the moon?
        Answer: The moon is 384,400 km away.
        Score: 5

        Question: How far away is the sun?
        Answer: I don't know
        Score: 0

        Your turn!

        Question : {question}
        """
        )

    def get_choose_prompt(self):
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                Use ONLY the following pre-existing answers to the user's question.

                Use the answers that have the highest score (more helpful) and favor the most recent ones.

                Return the sources of the answers as they are, do not change them.

                You must print out only one answer.
                
                Answer: {answers}
                """,
                ),
                ("human", "{question}"),
            ]
        )


class Chain:
    prompt = Prompts()

    def generate_llm(self, OpenAI_KEY):
        self.common_llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-3.5-turbo-0125",
            api_key=OpenAI_KEY,
        )

        self.choose_llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-3.5-turbo-0125",
            api_key=OpenAI_KEY,
            streaming=True,
            callbacks=[
                ChatCallbackHandler(),
            ],
        )

    def get_answers(self, inputs):
        documents = inputs["documents"]
        question = inputs["question"]
        answers_prompt = self.prompt.get_answers_prompt()
        answers_chain = answers_prompt | self.common_llm
        return {
            "question": question,
            "answers": [
                {
                    "answer": answers_chain.invoke(
                        {
                            "context": doc.page_content,
                            "question": question,
                        }
                    ).content,
                    "source": doc.metadata["source"],
                    "date": doc.metadata["lastmod"],
                }
                for doc in documents
            ],
        }

    def choose_answer(self, inputs):
        answers = inputs["answers"]
        question = inputs["question"]
        choose_prompt = self.prompt.get_choose_prompt()
        choose_chain = choose_prompt | self.choose_llm
        condensed = "\n\n".join(
            f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
            for answer in answers
        )
        return choose_chain.invoke({"question": question, "answers": condensed})

    def get_final_answer(self, question):
        final_chain = (
            {
                "documents": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self.get_answers)
            | RunnableLambda(self.choose_answer)
        )
        return final_chain.invoke(question)


@st.dialog("View Git repo & SiteGPT Code")
def view_code_info():
    st.write("Git repo \n\n https://github.com/Donghyeon-Shin/langchain-challenges")
    st.write(
        "QuizGPT Code \n\n https://github.com/Donghyeon-Shin/langchain-challenges/blob/master/pages/SiteGPT.py"
    )


st.set_page_config(
    page_title="Site GPT",
    page_icon="🔍",
)

st.title("Site GPT")

st.markdown(
    """
    Welcome!\n\n
    Use this chatbot to ask questions to an AI about Site!\n\n
    Please enter the site you want to search for.
    """
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:    
    llm_api = st.text_input("Input OpenAI API")
    site_Url = st.text_input("Input site url : .xml")
    view_info_button = st.button("Git & Code Info")
    if view_info_button:
        view_code_info()    

# https://developers.cloudflare.com/sitemap-0.xml

if llm_api.startswith("sk-") and site_Url.endswith(".xml"):
    retriever = get_retriever(site_Url, llm_api)
    if retriever != "Error":
        send_message("How can I help you?", "ai", False)
        paint_messages()
        question = st.chat_input("Ask anything about this site")
        if question:
            chain = Chain()
            send_message(question, "human")
            chain.generate_llm(OpenAI_KEY=llm_api)
            with st.chat_message("ai"):
                chain.get_final_answer(question)

else:
    st.session_state["messages"] = []
