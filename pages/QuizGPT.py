import streamlit as st
import time
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.retrievers import WikipediaRetriever

format_function = {
    "name": "formatting_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

quiz_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that is role playing as a teacher.
            Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
            Each question should have 4 answer, three of them must be incorrect and one should be correct.
            The problem is that there are two versions that are difficult and easy and they must be presented at the difficulty level desired by the user.
            You should MAKE 10 Questoins
            
            Use (o) to signal the correct answer.

            Question examples
            
            Question: What is the color of the occean?
            Answers: Red|Yellow|Green|Blue(o)

            Question: What is the capital or Georgia?
            Answers: Baku|Tbilisi(o)|Manila|Beirut
            
            Question: When was Avator released?
            Answers: 2007|2001|2009(o)|1998

            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(o)|Painter|Actor|Model

            Your turn!

            Context: {context}
            Difficulty : {difficulty}
            """,
        )
    ]
)


def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)


@st.cache_resource(show_spinner="Searching wikipedia...")
def search_wiki(topic):
    retriever = WikipediaRetriever(top_k_results=3)
    docs = retriever.get_relevant_documents(topic)
    return docs


def run_quiz_chain(llm_api, docs, difficulty):
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-0125",
        api_key=llm_api,
    ).bind(
        function_call={
            "name": "formatting_quiz",
        },
        functions=[
            format_function,
        ],
    )

    quiz_chain = quiz_prompt | llm
    context = format_docs(docs)
    return quiz_chain.invoke({"context": context, "difficulty": difficulty})


@st.cache_data(show_spinner="Making quiz....")
def get_quiz_json(_llm_api, _docs, difficulty):
    response = run_quiz_chain(_llm_api, _docs, difficulty)
    quiz_json = json.loads(response.additional_kwargs["function_call"]["arguments"])
    return quiz_json


@st.dialog("View Git repo & QuizGPT Code")
def view_code_info():
    st.write("Git repo \n\n https://github.com/Donghyeon-Shin/langchain-challenges")
    st.write("QuizGPT Code \n\n kk")


if not "start_status" in st.session_state:
    st.session_state["start_status"] = False

if not "quiz_json" in st.session_state:
    st.session_state["quiz_json"] = []

st.set_page_config(page_title="QuizGPT", page_icon="❓")

st.title("QuizGPT")

with st.sidebar:
    llm_api = st.text_input("Input OpenAI API")
    topic = st.text_input("Search Wikipedia")
    view_info_button = st.button("Git & Code Info")
    if view_info_button:
        view_code_info()
    docs = None
    if topic:
        docs = search_wiki(topic)
    if st.session_state["start_status"]:
        reset_button = st.button("Restart Quiz")
        if reset_button:
            st.cache_data.clear()
            st.session_state["start_status"] = False
            st.session_state["quiz_json"] = []
            st.rerun()

if not docs:
    st.markdown(
        """
        1. Enter keywords on the side.
        2. Choose quiz difficulty.
        3. Click quiz make button.
        4. Try to solve a problem!!
        """
    )
    st.session_state["start_status"] = False
    st.session_state["quiz_json"] = []
elif not llm_api.startswith("sk-"):
    st.warning("Please enter your OpenAI Key!", icon="⚠️")
else:
    quiz_json = None
    if not st.session_state["start_status"]:
        difficulty = None
        make_quiz_button = None

        left, right = st.columns(2, vertical_alignment="bottom")

        with left:
            difficulty = st.selectbox(
                "Choose quiz difficulty.", options=["Easy", "Hard"]
            )
        with right:
            make_quiz_button = st.button("Make quiz")

        if make_quiz_button:
            st.session_state["quiz_json"] = get_quiz_json(llm_api, docs, difficulty)
            st.session_state["start_status"] = True
            st.rerun()
    else:
        with st.form("quiz_form"):
            correct_cnt = 0
            questions = st.session_state["quiz_json"]["questions"]

            for question in questions:
                question["question"]
                st.write(question["question"])

                # Remove (o)
                for answer in question["answers"]:
                    answer["answer"] = answer["answer"].replace("(o)", "")

                value = st.radio(
                    "Select an answer",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    correct_cnt = correct_cnt + 1
                    st.success("Correct!")
                elif value != None:
                    st.error("Wrong")

            if correct_cnt == len(questions):
                st.session_state["start_status"] = False
                st.balloons()
                st.toast("You are perfect!", icon="🎉")
                time.sleep(2)
                st.rerun()

            st.form_submit_button("Submit")
