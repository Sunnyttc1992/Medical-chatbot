""" Logic for AI agent with knowledge base integration. """
from dotenv import load_dotenv
load_dotenv()
import os
from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.llms import ChatMessage, MessageRole

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=os.getenv("BEDROCK_KNOWLEDGE_BASE_ID","BEDROCK_KNOWLEDGE_BASE_ID"),
    retrieval_config={"vectorSearchConfiguration": {"k": 3}},
)

llm = OpenAI(model=os.getenv("OPENAI_MODEL","gpt-5.1"))

_knowledge_base_tool = QueryEngineTool.from_defaults(
    query_engine=RetrieverQueryEngine(retriever=retriever),
    name="Knowledge Base",
    description=("Useful for answering questions about Pediatric treatment and diagnosis.")
)

agent = ReActAgent(
    tools=[_knowledge_base_tool],
    llm=llm,
    system_prompt=(
        "You are a helpful AI assistant with access to a vector database of knowledge about Pediatric Discipline. "
        "When users ask questions about pediatric treatment and diagnosis,"
        "use the available tool to retrieve accurate information. "
        "Always provide clear and concise answers based on the retrieved information."
        "You must use English language for your responses and provide the answer in a concise manner."
    ),
)

async def get_agent_response(message, chat_history):
    """Get response from agent given user message and chat history."""
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(ChatMessage(role=MessageRole.USER, content=msg["content"]))
        elif msg["role"] == "assiantant":
            message.append(ChatMessage(role=MessageRole.ASSISTANT,content =msg["content"]))

    user_message = ChatMessage(role=MessageRole.USER, content=message)

    response = await agent.run(user_message, chat_history=messages)
    return str(response)
    