import os, tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

from src.vectordb.chroma import ChromaDb


def charge_split(tmp_file_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(tmp_file_path)
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    loader = loader.load_and_split(text_splitter=r_splitter)
    return loader

def conf_vector_db():
    embedding = AzureOpenAIEmbeddings(
            azure_endpoint=os.environ["AZURE_ENDPOINT2"],
            azure_deployment=os.environ["AZURE_EMBED_DEPLOY"],
            api_key=os.environ["OPENAI_EMBED_API_KEY"]
        )
    vectordb = ChromaDb(embedding=embedding)
    return vectordb

def process_pdf(vectordb, pdf_content, chunk_size, chunk_overlap):

    context = vectordb.get_collection('context')
    #save pdf's content in a temporal file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_content)
        tmp_file_path = tmp_file.name

    #split and charge pdf's content
    loader = charge_split(tmp_file_path, chunk_size, chunk_overlap)

    #uploading AzureOpenAIEmbeddings to ChromaDB
    context.from_documents(
        loader,
        embedding=vectordb.embedding,
        collection_name='context',
        persist_directory='./data'
        )
    context.persist()

    #erasing temporal file
    os.unlink(tmp_file_path)