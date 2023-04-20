from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.experimental import BabyAGI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from arxiv_api import ArxivAPIWrapper

embeddings_model = HuggingFaceEmbeddings()

# Initialize the vectorstore as empty
import faiss

embedding_size = 768
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

todo_prompt = PromptTemplate.from_template(
    "You are an planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
)
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
arxiv = ArxivAPIWrapper(top_k_results=3)

tools = [
    Tool(
        name="Arxiv",
        func=arxiv.run,
        description="useful for when you need to gather summaries and titles of research papers. Input: a query for research papers of the format 'cat:cs.<subject category> and <topic>' where subject category can be CL (for natural language processing), CV (for computer vision and pattern recognition), or LG (for general machine learning). Output: the top 3 paper titles and summaries matching the query."
    ),
    Tool(
        name="TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
    ),
    WriteFileTool(),
    ReadFileTool(),
]

prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["objective", "task", "context", "agent_scratchpad"],
)

llm = OpenAI(temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

OBJECTIVE = "write a detailed summary to a file on the latest trends in natural language processing and computer vision research."
# Logging of LLMChains
verbose = False
# If None, will keep on going forever
max_iterations = 4
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, task_execution_chain=agent_executor, verbose=verbose, max_iterations=max_iterations
)

baby_agi({"objective": OBJECTIVE})