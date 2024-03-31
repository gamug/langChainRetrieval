import chromadb, os, sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAI
from operator import itemgetter

class ChromaDb:
    
    def __init__(self, embedding=None):
        self.embedding = embedding
        self.persistent_client, memory, context = self.create_db()
        self.collections = {'memory': memory, 'context': context}
        self.qa_chain = self.set_completion()
    
    def create_db(self):
        persistent_client = chromadb.PersistentClient(os.path.join(sys.path[0], 'data/'))
        memory = persistent_client.get_or_create_collection('memory')
        context = persistent_client.get_or_create_collection('context')
        return persistent_client, memory, context
    
    def validate_collection(self, collection):
        collections = [col.name for col in self.persistent_client.list_collections()]
        assert collection in collections, f'''CollectionNotFound: no collection <{collection}> found in <{", ".join(collections)}>. Try first to create a collection with the name <{collection}>'''
    
    def get_collection(self, collection):
        self.validate_collection(collection)
        langchain_chroma = Chroma(
            client=self.persistent_client,
            collection_name=collection,
            embedding_function=self.embedding,
            persist_directory=os.path.join(sys.path[0], 'data/')
        )
        return langchain_chroma
    
    def get_collections(self, collections: list[str]):
        collections = tuple(self.get_collection(collection) for collection in collections)
        return collections

    def get_retriever(self, collection:str, k:int=7):
        langchain_chroma = self.get_collection(collection)
        langchain_chroma = langchain_chroma.as_retriever(search_kwargs={'k': k})
        return langchain_chroma

    def reset_collection(self, collection: str):
        self.validate_collection(collection)
        self.persistent_client.delete_collection(collection)
        self.collections[collection] = self.persistent_client.create_collection(collection)
        self.qa_chain = self.set_completion()
    
    def set_completion(self):
        #defining llm object
        llm = AzureOpenAI(
                api_key=os.environ['OPENAI_API_KEY'],
                azure_endpoint=os.environ["AZURE_ENDPOINT1"],
                api_version=os.environ["OPENAI_API_VERSION"],
                azure_deployment=os.environ["AZURE_GPT_DEPLOY"]
            )
        
        #providing prompt template 
        template = '''You're an expert designing innovation models applied to social topics. We passs to you as context some useful documents.\n
                    We also pass chat history to support customer's advanced interactions.

                    Context data: {context}
                    Chat history: {history}

                    provide the answer to the question {query} in the languange {language}.
                    '''
        prompt = ChatPromptTemplate.from_template(template)

        #connecting the chain
        qa_chain = (
            {
                'context': itemgetter('query') | self.get_retriever('context'),
                'query': itemgetter('query'),
                'language': itemgetter('language'),
                'history': itemgetter('query') | self.get_retriever('memory')
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return qa_chain