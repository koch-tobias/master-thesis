import openai
def init_openai():
    openai.api_type = "azure"
    openai.api_base = "https://convaip-sbx-openai.openai.azure.com/"
    # openai.api_version = "2022-12-01" # For GPT3.0
    openai.api_version = "2023-03-15-preview" # For GPT 3.5
    openai.api_key = '9e6fa24631f54cf58866766bd31a2bff' #os.getenv("OPENAI_API_KEY")