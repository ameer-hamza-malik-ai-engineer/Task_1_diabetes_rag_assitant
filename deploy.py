import uvicorn
from retrieval_pipeline import query_rag
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def hello_world():
    return {"message": "Please visit /docs for the API documentation and testing interface."}


@app.post("/ask")
def query_endpoint(query: QueryRequest):
    answer, docs_and_scores = query_rag(query.query)
    sources = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": round(score, 4),
        }
        for doc, score in docs_and_scores
    ]
    return {
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    port = 8000

    uvicorn.run("deploy:app", host="0.0.0.0", port=port)