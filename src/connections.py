import os

def set_connections():
    os.environ["OPENAI_API_TYPE"] = '***'
    os.environ["OPENAI_API_KEY"] = '******'
    os.environ["OPENAI_EMBED_API_KEY"] = '******'
    os.environ["AZURE_ENDPOINT1"] = '********'
    os.environ["AZURE_ENDPOINT2"] = '******'
    os.environ["AZURE_GPT_DEPLOY"] = '*******'
    os.environ["AZURE_EMBED_DEPLOY"] = '******'
    os.environ["OPENAI_API_VERSION"] = '2023-12-01-preview'
