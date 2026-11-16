"""LangGraph workflow: rewrite -> retrieve -> (relevant? generate : no_answer)."""
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from .config import GEMINI_MODEL, IMAGE_MIN_SIM, TEXT_MIN_SIM, TOP_K
from .retrieval import retrieve

NOT_FOUND_MESSAGE = "I could not find anything relevant to this question in the document(s)."
MAX_HISTORY_MESSAGES = 6  # last 3 question/answer pairs


class RAGState(TypedDict, total=False):
    query: str  # what the user typed
    history: list  # [{"role": "user"/"assistant", "content": str}]
    standalone_query: str  # query rewritten to make sense without the chat
    retrieved: list  # [(Document, similarity, used)]
    docs: list[Document]  # documents that passed the relevance threshold
    answer: str


def get_llm():
    return init_chat_model(f"google_genai:{GEMINI_MODEL}", max_retries=3, timeout=180)


def _text_of(content) -> str:
    """Gemini 3 returns a list of content blocks; keep only the text."""
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return content


def build_message(query: str, docs: list[Document], image_store: dict) -> HumanMessage:
    content = [{"type": "text", "text": f"Question: {query}\n\nContext:\n"}]

    text_docs = [d for d in docs if d.metadata.get("type") == "text"]
    image_docs = [d for d in docs if d.metadata.get("type") == "image"]

    if text_docs:
        ctx = "\n\n".join(
            f"[{d.metadata.get('source', '')} p.{d.metadata['page'] + 1}]: {d.page_content}"
            for d in text_docs
        )
        content.append({"type": "text", "text": f"Text excerpts:\n{ctx}\n"})

    for d in image_docs:
        b64 = image_store.get(d.metadata.get("image_id"))
        if b64:
            content.append(
                {
                    "type": "text",
                    "text": f"\n[Image from {d.metadata.get('source', '')} p.{d.metadata['page'] + 1}]:\n",
                }
            )
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )

    content.append(
        {
            "type": "text",
            "text": (
                "\n\nAnswer the question using only the provided text and images. "
                'If they do not contain the answer, reply exactly: '
                '"I could not find this in the document." Do not use outside knowledge.'
            ),
        }
    )
    return HumanMessage(content=content)


def build_graph(vector_store, image_store: dict, llm):
    def rewrite_node(state: RAGState):
        query = state["query"]
        history = (state.get("history") or [])[-MAX_HISTORY_MESSAGES:]
        if not history:  # first question: nothing to resolve, skip the LLM call
            return {"standalone_query": query}

        convo = "\n".join(
            f"{m['role'].capitalize()}: {m['content'][:500]}" for m in history
        )
        prompt = (
            "Given the conversation so far and a follow-up question, rewrite the "
            "follow-up as a standalone question that can be understood without the "
            "conversation. Keep names, numbers and technical terms. If it is already "
            "standalone, return it unchanged. Output only the question.\n\n"
            f"Conversation:\n{convo}\n\nFollow-up question: {query}\n\nStandalone question:"
        )
        try:
            rewritten = _text_of(llm.invoke(prompt).content).strip()
        except Exception:
            rewritten = ""
        return {"standalone_query": rewritten or query}  # fall back to the original

    def retrieve_node(state: RAGState):
        retrieved, kept = [], []
        for doc, sim in retrieve(vector_store, state["standalone_query"], TOP_K):
            threshold = TEXT_MIN_SIM if doc.metadata.get("type") == "text" else IMAGE_MIN_SIM
            used = sim >= threshold
            retrieved.append((doc, sim, used))
            if used:
                kept.append(doc)
        return {"retrieved": retrieved, "docs": kept}

    def route(state: RAGState):
        return "generate" if state["docs"] else "no_answer"

    def generate_node(state: RAGState):
        message = build_message(state["standalone_query"], state["docs"], image_store)
        return {"answer": _text_of(llm.invoke([message]).content)}

    def no_answer_node(state: RAGState):
        return {"answer": NOT_FOUND_MESSAGE}

    builder = StateGraph(RAGState)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("no_answer", no_answer_node)

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_conditional_edges(
        "retrieve", route, {"generate": "generate", "no_answer": "no_answer"}
    )
    builder.add_edge("generate", END)
    builder.add_edge("no_answer", END)
    return builder.compile()