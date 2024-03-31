from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.connections import set_connections
from server.objects import Question
from src.langchain_utils import (
    process_pdf,
    conf_vector_db
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

set_connections()
vectordb = conf_vector_db()

@app.get("/")

def index():
    return{"message": "All working well from this side!!"}

@app.post("/process-document/")
async def process_document(pdf: UploadFile = File(...)):

    #verifing pdf format in file
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail=f"File format not supported: expected <application/pdf> got <{pdf.content_type}>")

    #reading pdf's content
    pdf_content = await pdf.read()

    # process pdf (spliting, vectorizing and storing)
    process_pdf(vectordb, pdf_content)

    return {"response": "DocumentProcessedSuccessfully"}

@app.post("/make-question/")
async def make_question(question: Question):

    #defining basic variables
    memory = vectordb.get_collection('memory')
    question_str = question.question
    print(question_str)
    
    #asking questions to ChatGPT
    answer = vectordb.qa_chain.invoke({'query': question_str, 'language': 'spanish'})
    memory.add_texts(['\n\nuser_saids:\n'+question_str, '\n\nyou answer:\n'+answer])
    memory.persist()

    return {"response": answer}

@app.post("/delete-context/")
async def delete_context():
    vectordb.reset_collection('context')
    return {'response': 'ContextSuccessfullyCleaned'}

@app.post("/delete-memory/")
async def delete_memory():
    vectordb.reset_collection('memory')
    return {'response': 'MemorySuccessfullyCleaned'}