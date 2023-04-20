from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI

from langchain.experimental import AutoGPT
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

from arxiv_api import ArxivAPIWrapper

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings

# Initialize the vectorstore as empty
import faiss

embeddings_model = HuggingFaceEmbeddings()
embedding_size = 768 #1536- openai
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

## Task Creation
# todo_prompt = PromptTemplate.from_template(
#     "You are an planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
# )
# todo_chain = LLMChain(llm=OpenAI(temperature=0.1), prompt=todo_prompt)
arxiv = ArxivAPIWrapper(top_k_results=3)

with open('arxiv_api_spec.txt', 'r') as f:
    api_doc = f.read()


"""" old arxiv tool description
useful for when you need to gather summaries and titles of research papers. Input: a query for research papers that follows the format 'cat:cs.<subject category> and <keywords>' where subject category can be CL (for natural language processing), CV (for computer vision and pattern recognition), or LG (for general machine learning). Output: the top 3 paper titles and summaries matching the query.
"""

# "useful for when you need to gather summaries and titles of research papers. Input: a query for research papers that follows the api documentation and where the subject category is cat:cs.<CL, CV, or LG> for natural language processing, computer vision, or machine learning. Output: the top 3 paper titles and summaries matching the query. Here is the api documentation: " + api_doc

tools = [
    Tool(
        name="Arxiv",
        func=arxiv.run,
        description="useful for when you need to gather summaries and titles of research papers. Input: a query for research papers of the format 'cat:cs.<subject category> and <topic>' where subject category can be CL (for natural language processing), CV (for computer vision and pattern recognition), or LG (for general machine learning). Output: the top 3 paper titles and summaries matching the query."
    ),
    WriteFileTool(),
    ReadFileTool(),
]

agent = AutoGPT.from_llm_and_tools(
    ai_name="Koda",
    ai_role="Research Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    memory=vectorstore.as_retriever()
)
# Set verbose to be true
agent.chain.verbose = False
agent.run(["write a detailed summary on the latest trends in natural language processing and computer vision research."])
