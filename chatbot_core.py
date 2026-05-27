from docling.document_converter import DocumentConverter
from bs4 import Tag, BeautifulSoup
from langchain_text_splitters import HTMLSectionSplitter, RecursiveCharacterTextSplitter
import time
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import Literal
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.messages import HumanMessage
from langchain_core.messages import convert_to_messages
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableConfig
import os



#Create a retriever tool using the @tool decorator:
@tool
def retrieve_blog_posts(query: str, config: RunnableConfig) -> str:
    """Search and return information about Headset Profile (HSP)."""
    retriever = config["configurable"]["retriever"]
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

retriever_tool = retrieve_blog_posts
#Generate query
response_model = init_chat_model("Qwen/Qwen3.5-0.8B", 
    model_provider="huggingface",
    temperature=0.7,
    max_tokens=32768
    # max_tokens=1024,
)
response_model.bind_tools([retriever_tool])


grader_model = init_chat_model("Qwen/Qwen3.5-0.8B", 
    model_provider="huggingface",
    temperature=0.7,
    # max_tokens=1024,
)


def pdf2split(pdf_filepath):
    converter = DocumentConverter()
    result = converter.convert(pdf_filepath)

    temp_html = "temp_output.html"
    result.document.save_as_html(filename=temp_html)

    with open(temp_html, "r", encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")
    body = soup.body

    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
    ]

    html_splitter = HTMLSectionSplitter(headers_to_split_on)
    sections = html_splitter.split_text(str(body))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    final_chunks = []
    for section in sections:
        chunks = text_splitter.split_text(section.page_content)
        final_chunks.extend(chunks)

    return final_chunks

def create_retriever(pdf_filepath):
    html_header_splits_as_string = pdf2split(pdf_filepath)
    html_header_splits = [Document(page_content=split_string) for split_string in html_header_splits_as_string]
    #Create a retriever tool
    max_emb_batch_size = 32
    print("creating embedding model")
    embeddings = OpenAIEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
        base_url=os.getenv("EMBEDDINGS_URL", "http://localhost:8080/v1"),
    api_key="bngjfjj")
    print("embeddings model created")

    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    print("vector store created")
    N = len(html_header_splits)
    uuids = [str(uuid4()) for _ in range(N)]
    print(f"Indexing {len(html_header_splits)} items ...")

    for start_idx in range(0, N, max_emb_batch_size):
        end_idx = min(N, start_idx + max_emb_batch_size)
        start_time = time.time()
        vector_store.add_documents(documents=html_header_splits[start_idx: end_idx], ids=uuids[start_idx: end_idx])
        stop_time = time.time()
        print(f"Indexed {end_idx-start_idx} items in {stop_time-start_time} seconds")
    vectorstore = InMemoryVectorStore.from_documents(
        documents=html_header_splits, embedding=HuggingFaceEmbeddings()
    )
    retriever = vectorstore.as_retriever()

    return retriever



def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        response_model.invoke(state["messages"])
    )
    return {"messages": [response]}

def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"
    
def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}

def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}



class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


#Grade Document
GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

#rewrite questions
REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

#generate answer
GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)








    
#Run this with irrelevant documents in the tool response: (check code getting NotImplementedError: Pydantic schema is not supported for function calling)

def create_graph():
    #assemble the graph
    workflow = StateGraph(MessagesState)

    # Define the nodes we will cycle between
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)
    workflow.add_edge(START, "generate_query_or_respond")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        # Assess LLM decision (call `retriever_tool` tool or respond to the user)
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    

    return workflow.compile()






# -------- Main Program --------
def main():
    #compile
    graph = create_graph()
    
    pdf_path = input("Enter PDF file path: ").strip()

    if not pdf_path:
        print("Invalid path!")
        return

    print("Processing PDF...")
    retriever = create_retriever(pdf_path)

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Retrieve relevant chunks
        

        for chunk in graph.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{query}",
                    }
                ]
            },
            config={"configurable": {"retriever": retriever}},
        ):
            for node, update in chunk.items():
                print("Update from node", node)
                update["messages"][-1].pretty_print()
                print("\n\n")


if __name__ == "__main__":
    main()
