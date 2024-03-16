import os, tempfile
from typing import Dict, Any
from operator import itemgetter

from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import RetrievalQA
from langchain_openai import AzureOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationSummaryBufferMemory


def charge_split(tmp_file_path):
    loader = PyPDFLoader(tmp_file_path)
    chunk_size = 700 #jugar con estos valores
    chunk_overlap = 150 #jugar con estos valores
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
    vectordb = Chroma(
        'BID_POC',
        embedding,
        persist_directory='./data'
    )
    vectordb.persist()
    return vectordb, embedding

def process_pdf(vectordb, embedding, pdf_content):

    # Guardar el contenido del PDF en un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_content)
        tmp_file_path = tmp_file.name

    # Cargar y dividir el texto del PDF
    loader = charge_split(tmp_file_path)

    # Configuramos AzureOpenAIEmbeddings y Chroma
    vectordb = vectordb.from_documents(loader, embedding=embedding, collection_name='BID_POC', persist_directory='./data')
    vectordb.persist()

    # Eliminar el archivo temporal
    os.unlink(tmp_file_path)

class AnswerConversationBufferMemory(ConversationTokenBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(inputs,{'response': outputs['result']})
    
def set_completion(vectordb):

    # Configuramos RetrievalQA
    llm = AzureOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            azure_endpoint=os.environ["AZURE_ENDPOINT1"],
            api_version=os.environ["OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_GPT_DEPLOY"]
        )
    template = '''Answer the question based on the following context: {context}

                You also can find usefull the chat history: {history}

                Question: {query}

                Answer in the following language {language}
                '''
    prompt = ChatPromptTemplate.from_template(template)
    memory = ConversationSummaryBufferMemory(llm=llm, return_messages=True, max_token_limit=500)
    # memory.load_memory_variables({})
    qa_chain = (
        {
            'context': itemgetter('query') | vectordb.as_retriever(search_kwargs={'k': 7}),
            'query': itemgetter('query'),
            'language': itemgetter('language'),
            # "history": RunnablePassthrough()
            'history': RunnablePassthrough.assign(
                history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=vectordb.as_retriever(search_kwargs={'k': 7}),
    #     return_source_documents=True,
    #     memory=AnswerConversationBufferMemory(llm=llm, return_messages=True, max_token_limit=500)
    # )
    return qa_chain, memory