import os

def set_connections():
    os.environ["OPENAI_API_TYPE"] = 'api_key'
    os.environ["OPENAI_API_KEY"] = '6286ae4863e54163bc669839b2531f1a'
    os.environ["OPENAI_EMBED_API_KEY"] = 'bb3d1d96f030475e918408f131285649'
    os.environ["AZURE_ENDPOINT1"] = 'https://oai-llm-poc-ecp-sbx.openai.azure.com/'
    os.environ["AZURE_ENDPOINT2"] = 'https://oai-llmca-poc-ecp-sbx.openai.azure.com/'
    os.environ["AZURE_GPT_DEPLOY"] = 'Anie'
    os.environ["AZURE_EMBED_DEPLOY"] = 'genia-embedding'
    os.environ["OPENAI_API_VERSION"] = '2023-12-01-preview'