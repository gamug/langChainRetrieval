import sys, re, os
from operator import itemgetter

from langchain_core.runnables import RunnableLambda
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.connections import set_connections
from server.objects import Documento, Pregunta
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

@app.post("/procesar-documento/")
async def procesar_documento(pdf: UploadFile = File(...)):

    # Verificamos que el archivo sea un PDF
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="El archivo no es un PDF")

    # Leemos el contenido del PDF
    contenido_pdf = await pdf.read()

    # Procesamos el PDF si no se ha procesado antes

    process_pdf(vectordb, embedding, contenido_pdf)

    return {"mensaje": "Documento PDF procesado exitosamente"}

@app.post("/hacer-pregunta/")
async def hacer_pregunta(pregunta: Pregunta):
    pregunta_str = pregunta.pregunta
    print(pregunta)

    # Hacemos preguntas a ChatGPT
    respuesta = qa_chain.invoke(
        {'query': pregunta_str,
        #  'history': RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
         'language': 'spanish'}
    )
    memory.save_context({'input': pregunta_str}, {'output': respuesta})
    return {"respuesta": respuesta}

if __name__=='__main__':
    # Configuración de CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # Origen permitido para las solicitudes CORS
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )