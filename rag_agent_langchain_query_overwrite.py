import bs4
import os

import dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_tool_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import InjectedState

#----------------blog加载与切分----------------
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
#------------------------embedding和向量数据库--------------------------------
from langchain_ollama import OllamaEmbeddings, ChatOllama  # pip install langchain-ollama
from langgraph.prebuilt.tool_node import ToolCallRequest
from pydantic.v1 import validator

embedding = OllamaEmbeddings(
    model="bge-m3:latest",
    base_url="http://localhost:11434",
)
#获取向量数据库
from langchain_chroma import Chroma
vector_store = Chroma(
    collection_name="rag_docs",
    embedding_function=embedding,
    persist_directory="./chroma_db",
    collection_metadata={"hnsw:space": "cosine"}
)

# 有数据就跳过，没有才写入
if vector_store._collection.count() == 0:
    print("首次写入数据...")
    _ = vector_store.add_documents(documents=all_splits)
else:
    print(f"已有 {vector_store._collection.count()} 条数据，跳过写入")
#----------------------rerank--------------------------
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)


def rerank_documents(query: str, docs: list, top_k: int = 2) -> list:
    """对检索到的文档进行重排序，只返回最相关的 top_k 条"""
    print("*************rerank**********************")
    if not docs:
        return []

    # 构建查询-文档对
    pairs = [[query, doc.page_content] for doc in docs]

    # 计算每个文档的精确相关性分数
    scores = reranker.predict(pairs)

    # 按分数降序排列，取 top_k
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    # ✅ 添加调试打印
    print(f"Rerank 查询: {query}")
    print(f"输入文档数: {len(docs)}, 输出 top_k: {top_k}")
    print("-" * 40)
    for i, (score, doc) in enumerate(scored_docs):
        prefix = "✅" if i < top_k else "❌"
        print(f"{prefix} 排名{i + 1}: 得分 {score:.4f}")
        print(f"   内容预览: {doc.page_content[:100]}...")
        print()
    #rerank：相关性分数，越大越好，所以 score > 0.5 保留
    # 只返回得分大于 0.5 的文档，且不超过 top_k 条
    filtered_docs = [doc for score, doc in scored_docs if score > 0.5]
    print(f"最终返回文档得分大于 0.5：{filtered_docs}")
    return filtered_docs[:top_k]
#----------------------查询重写----------------------
from langchain_core.prompts import ChatPromptTemplate
#todo本地模型有点弱,不用了已使用云端model_agent
model_ollama = ChatOllama(
    model="deepseek-r1:8b",
    base_url="http://localhost:11434",
    temperature=0.1,
    think=False,
)
#todo：配置文件地址
dotenv.load_dotenv('D:\PythonProject\langchainaitest\lac\.env')#修改你的配置文件地址
model_agent = init_chat_model(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model_provider="openai",
    model="qwen3.5-35b-a3b",
    temperature=0.1,
    timeout=120,
)
from pydantic import BaseModel, Field

class RetrieveInput(BaseModel):
    """检索工具的参数"""
    question: str = Field(description="英文检索查询语句")

    @validator("question")
    def must_be_english(cls, v):  # ← 去掉 @classmethod
        """强制校验：不允许包含中文字符"""
        if any('\u4e00' <= char <= '\u9fff' for char in v):
            raise ValueError(f"查询语句必须为英文，检测到中文字符: {v}")
        return v


# 1. 查询重写提示词（包含对话历史）
query_rewrite_prompt = ChatPromptTemplate.from_template(
    """根据对话历史和用户问题，生成3个不同角度的检索查询，查询要使用英文，以提高检索质量。

对话历史:
{history}

用户问题: {question}

改写后的查询（每行一个）:"""
)
# 2. 提取对话历史
def get_chat_history(messages: list) -> str:
    """从消息列表中提取对话历史"""
    history = []
    for msg in messages:
        if msg.type == "human":
            history.append(f"用户: {msg.content}")
        elif msg.type == "ai":
            history.append(f"AI: {msg.content}")
    print("===========历史对话============")
    print(history)
    return "\n".join(history)
# 3.生成多个查询

def generate_queries(question: str,message:list) -> list[str]:
    history = get_chat_history(message)
    response=model_agent.invoke(
        query_rewrite_prompt.format(history=history, question=question),
    )
    print("===========多个查询============")
    print(response)
    queries=[q.strip() for q in response.content.split("\n") if q.strip()]
    seen=set()
    unique_queries = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)

    # 2. 过滤掉和原始问题一模一样的改写
    unique_queries = [q for q in unique_queries if q != question]

    return unique_queries


from langchain_core.tools import tool
from typing import Annotated, Callable
#------------------------中间件-----------------------
@wrap_tool_call
def translate_query(
        request: ToolCallRequest,
        handler: Callable[[ModelRequest], ModelResponse],
):
    """在检索工具执行前，自动将中文查询翻译成英文"""
    print("********translate_query中间件执行**************")
    if request.tool_call["name"]=="retrieve_with_query_rewrite":
        question = request.tool_call["args"].get("question")
        if any('\u4e00' <= char <= '\u9fff' for char in question):
            print(f"问题是：{question}")
            print(f"发现：{question}不是英文会影响检索")
            translate_prompt = f"将以下中文翻译成英文，只输出英文翻译结果:\n{question}"
            english_q = model_agent.invoke(translate_prompt).content.strip()
            request.tool_call["args"]["question"] = english_q

    return handler(request)
#----------------------------检索生成----------------------------
@tool(args_schema=RetrieveInput, response_format="content_and_artifact")
def retrieve_with_query_rewrite(
        question: str,
        state: Annotated[dict, InjectedState]  # ← 用InjectedState注解
) -> tuple:
    """根据问题进行查询重写并检索相关信息，返回检索结果。"""

    # 直接从state里拿消息历史
    messages = state["messages"]

    # 查询重写（传入真实的对话历史）
    queries = generate_queries(question, messages)
    #设置评分阈值（COSINE距离，越小越相似）
    SCORE_THRESHOLD = 0.5

    all_docs = []
    seen = set()
    query_all=[question] + queries

    #带评分与阈值的检索
    for i,query in enumerate(query_all, 1):
        print(f"----------问题{i}---------------")
        print(f"当前问题：{query}")
        doc_with_scores=vector_store.similarity_search_with_score(query,k=3)
        for doc,score in doc_with_scores:
            if score > SCORE_THRESHOLD:
                print(f"  ❌ 跳过（得分{score:.4f} > 阈值{SCORE_THRESHOLD}）")
                continue
            print(f"  ✅ 保留（得分{score:.4f}）")
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)
    #rerank排序，衡量文档和用户真实意图的相关性
    all_docs = rerank_documents(question, all_docs, top_k=3)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in all_docs
    )
    return serialized, all_docs



#----------------------创建agent-----------------
prompt = (
    "你是一个专业的AI助手，具备检索能力。"
    "请使用检索工具查找相关信息回答用户问题。"
    "如果没有找到相关信息，请如实说明。"
    "当使用retrieve_with_query_rewrite工具时，需要将用户问题翻译成英文再传入参数，提高检索质量"
)
# Agent直接用这个tool
tools = [retrieve_with_query_rewrite]
agent = create_agent(model_agent, tools=tools, system_prompt=prompt,middleware=[translate_query])
# 模拟多轮对话历史
messages = [
    {"role": "user", "content": "什么是AI Agent？"},
    {"role": "assistant", "content": "AI Agent是一种能够自主感知环境、做出决策并采取行动的智能系统。"},
    {"role": "user", "content": "它和普通的LLM有什么区别？"},
    {"role": "assistant", "content": "普通LLM只能被动回答问题，而Agent可以主动调用工具、规划步骤、完成复杂任务。"},
    {"role": "user", "content": "那任务分解在其中扮演什么角色？中文回答"},
]

print("=== 多轮对话 Agent 回答 ===")
for step in agent.stream(
    {"messages": messages},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
