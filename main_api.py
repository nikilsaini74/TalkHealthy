import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from pydantic import BaseModel

from main import get_llm_instance
from scripts.app_environment import chromaDB_manager
from scripts.app_qa_builder import process_database_question, process_query
from main_ingest import ingest

sys.path.append(str(Path(sys.argv[0]).resolve().parent.parent))

app = FastAPI()

load_dotenv()

# Initialize a chat history list
chat_history = []


class ErrorResponse(BaseModel):
    status_code: int
    error: str


class QueryBody(BaseModel):
    database_name: str
    question: str


class LLM:
    def __init__(self):
        self.instance = None

    def get_instance(self):
        if not self.instance:
            self.instance = get_llm_instance()
        return self.instance


llm_manager = LLM()


@app.on_event("startup")
async def startup_event():
    llm_manager.get_instance()
    run_ingest()


def get_llm():
    return llm_manager.get_instance()


def list_of_collections(database_name: str):
    client = chromaDB_manager.get_client(database_name)
    return client.list_collections()


@app.get("/")
async def root():
    return {"ping": "pong!"}


def run_ingest():
    ingest()


@app.get('/databases')
async def get_database_names_and_collections():
    base_dir = "./db"
    try:
        database_names = \
            sorted([name for name in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, name))])

        database_info = []
        for database_name in database_names:
            collections = list_of_collections(database_name)
            database_info.append({
                'database_name': database_name,
                'collections': collections
            })

        return database_info
    except Exception as e:
        return ErrorResponse(status_code=500, error=str(e))


@app.post('/query')
async def query_documents(body: QueryBody, llm=Depends(get_llm)):
    database_name = body.database_name
    question = body.question

    try:

        seeking_from = database_name + '/' + database_name
        print(f"\n\033[94mSeeking for answer from: [{seeking_from}]. May take some minutes...\033[0m")
        qa = process_database_question(database_name, llm, database_name)
        answer, docs = process_query(qa, question, chat_history, chromadb_get_only_relevant_docs=False, translate_answer=False)

        source_documents = []
        for doc in docs:
            document_page = doc.page_content.replace('\n', ' ')

            source_documents.append({
                'content': document_page,
                'link': doc.metadata['source']
            })

        response = {
            'answer': answer,
            'source_documents': source_documents
        }
        return response
    except Exception as e:
        return ErrorResponse(status_code=500, error=str(e))


# commented out, because we use web UI
if __name__ == "__main__":
    import uvicorn

    host = '0.0.0.0'
    port = 8080
    print(f"TalkHealthy API is now available at http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port)
