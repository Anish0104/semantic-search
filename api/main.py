# API entry point
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Semantic Search API is running"}

@app.post("/search")
def search(query: str):
    return {"query": query, "results": []}
