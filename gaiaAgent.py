from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import whisper

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

from pydantic import Field
from langchain.tools import BaseTool, Tool, tool
from smolagents import CodeAgent, OpenAIServerModel

from langchain_community.vectorstores import Chroma
from langchain_community.tools.riza.command import ExecPython
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores.utils import filter_complex_metadata

import pandas as pd
import gradio as gr

from tabulate import tabulate  # pragma: no cover – fallback path


import re
import os
import getpass
import requests


# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
    
_set_env("RIZA_API_KEY")



def execute_python_code(query: str):
    """
    This function executes Python code in the attachment and returns the output of the code.
    """
    exec_python = ExecPython()
    code = attachment.text
    codeResults = exec_python.invoke(code)

    answer = llm.invoke(f"Given that the output of the program is {codeResults} please answer the question {query}.")
    return answer.content

def search_wikipedia(query: str):
    """Searches Wikipedia for the query."""
    wiki_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    myPrompt = "Given the context {context} please answer the question {question}."
            
    wiki_answer = "\n\n---\n\n".join(
        [
            f'doc.page_content'
            for doc in wiki_docs
        ]
    )
    
    myPrompt = myPrompt.format(context=wiki_answer, question=query)
    gptAns = llm.invoke(myPrompt)
    return gptAns


def youtube_transcript_search(query:str):
    """Searches Youtube for the query."""

    video_link = re.search("https://www.youtube.com/watch\?v=(.*)", mymessages[0].content)
    video_id = video_link.group(1)[0:11]
    loader = YoutubeLoader(video_id, add_video_info=False)
    docs = loader.load()
    docs = filter_complex_metadata(docs)

    ## Need to search embeddings instead of directly since OpenAI can't currently view YoutTube videos.
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings)
    answer =  vectorstore.similarity_search(query)[0].page_content

    return answer

def transcribe_audio(query: str):
    "transcribes audio from a file"
    with open('audio.mp3', 'wb') as f:
        f.write(attachment.content)

    result = whisperModel.transcribe("./audio.mp3")["text"]
    return result

def read_and_process_excel(query: str):
    "reads and processes an excel file using a code interpreter"

    result = pd.read_excel(attachment.content)  


    markdown = None

    if hasattr(result, "to_markdown"):
        markdown = result.to_markdown(index=False)
    else:
        markdown =  tabulate(result, headers="keys", tablefmt="github", showindex=False)

    codeAgent = CodeAgent(
            model=OpenAIServerModel(model_id="gpt-4o-mini"),
            tools=[],
            add_base_tools=True,
            additional_authorized_imports=['pandas','numpy','csv','subprocess', 'io']
    )

    myCodeResults = codeAgent(f"Please answer the following question '{query}'.  Put the data in markdown into a dataframe before performing the calculation.  Please provide the answer only, no other text.\n\nMarkdown: "+ markdown)

    finalAnswer = llm.invoke(f"Given that the result of executing the code is '{myCodeResults}' please provide your final answer using a single number as your final answer without a dollar sign but with two decimal places.")

    return finalAnswer.content

tools = [execute_python_code, search_wikipedia, youtube_transcript_search, transcribe_audio, read_and_process_excel]

# Define LLM with bound tools
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)
whisperModel = whisper.load_model("tiny")

# System message
sys_msg = SystemMessage(content="You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.  Use only the tools provided to you - do not use any other tools.  You may only process attachments using the provided tools.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")


checkpointer = MemorySaver()

# Compile graph
graph = builder.compile(checkpointer=checkpointer)


config = {"configurable": {"thread_id": "1"}}



api_url = DEFAULT_API_URL
files_url = f"{api_url}/files/"

def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        files_url = f"{api_url}/files/"

        global attachment 
        attachment = requests.get(files_url + task_id, timeout=30)
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            global mymessages 
            mymessages= [HumanMessage(content=question_text)]
            results = graph.invoke({"messages": mymessages}, config)
            messages = results["messages"]
            with_final_answer= messages[-1].content

            try: 
                submitted_answer = re.search(r"FINAL ANSWER: (.*)", with_final_answer).group(1)
            except: 
                submitted_answer = with_final_answer
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)