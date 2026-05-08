import os
import re
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient
from embedding_model import get_embedding_model

load_dotenv()

MODEL = "moonshotai/Kimi-K2.6"
LLM_INFERENCE_PROVIDER = "fireworks-ai"


def clean_response(text: str) -> str:
    """Normalize escaped newlines, strip Markdown syntax, and clean whitespace."""
    # Normalize escaped sequences from model output
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    # Remove ATX headers (## Heading → Heading)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers (**text** / *text* / __text__ / _text_)
    text = re.sub(r"\*{1,2}([^*\n]+)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}([^_\n]+)_{1,2}", r"\1", text)
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def llm(query, relevant_docs):
    client = InferenceClient(
        provider=LLM_INFERENCE_PROVIDER,
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"
    )

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable medical assistant. "
                    "Answer questions clearly and concisely using only the provided context. "
                    "Format your response using proper Markdown: use bullet points, bold key terms, "
                    "and paragraph breaks where appropriate. "
                    "Do not reveal chain-of-thought reasoning. Return only the final answer. "
                    "If you don't know the answer based on the provided context, say you don't know instead of making up an answer."
                )
            },
            {
                "role": "user",
                "content": prompt.format(context=format_docs(relevant_docs), question=query)
            }
        ],
        max_tokens=800,
        temperature=0.2,
        extra_body={"thinking": {"type": "disabled"}}
    )
    return clean_response(completion.choices[0].message.content)


def query_rag(query):
    persistent_directory = "db/chroma_db"

    embedding_model = get_embedding_model()

    db = Chroma(collection_name="langchain", embedding_function=embedding_model, persist_directory=persistent_directory,
    collection_metadata={"hnsw:space": "cosine"})

    k = int(os.getenv("K", 5))

    docs_and_scores = db.similarity_search_with_relevance_scores(query, k=k, score_threshold=0.3)
    relevant_docs = [doc for doc, _ in docs_and_scores]
    answer = llm(query, relevant_docs)

    return answer, docs_and_scores
