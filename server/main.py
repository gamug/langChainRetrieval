import sys, re, os
from operator import itemgetter

from langchain_core.runnables import RunnableLambda
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.connections import set_connections
from server.objects import Question
from src.langchain_utils import (
    process_pdf,
    set_completion,
    conf_vector_db
)


app = FastAPI()

set_connections()
vectordb, embedding = conf_vector_db()
qa_chain, memory = set_completion(vectordb)

@app.get("/")

def index():
    return{"message": "All working well from this side!!"}

@app.post("/process-document/")
async def process_document(pdf: UploadFile = File(...)):

    # Verificamos que el archivo sea un PDF
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="El archivo no es un PDF")

    # Leemos el contenido del PDF
    pdf_content = await pdf.read()

    # Procesamos el PDF si no se ha procesado antes

    process_pdf(vectordb, embedding, pdf_content)

    return {"response": "DocumentProcessedSuccessfullyDocumento"}

@app.post("/make-question/")
async def make_question(question: Question):
    question_str = question.question
    print(question_str)

    memory.add_user_message(question_str)
    # Hacemos preguntas a ChatGPT
    answer = qa_chain.invoke({'query': question_str, 'language': 'spanish', 'history': memory.messages})

    memory.add_user_message(answer)

    return {"response": answer}

@app.post("/delete-vectordb/")
async def delete_vectordb():
    ids = vectordb.get()['ids']
    print(ids)
    vectordb.delete(ids)
    return {'response': 'VectorDBSuccessfullyCleaned'}

@app.post("/delete-memory/")
async def delete_vectordb():
    memory.clear()
    return {'response': 'MemorySuccessfullyCleaned'}


if __name__=='__main__':
    # Configuración de CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # Origen permitido para las solicitudes CORS
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )