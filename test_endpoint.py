from langchain_community.llms.azureml_endpoint import (
    AzureMLEndpointApiType,
    AzureMLOnlineEndpoint,
    CustomOpenAIContentFormatter,
)

URI = None
KEY = None
 
llm = AzureMLOnlineEndpoint(
    endpoint_url=URI,
    endpoint_api_type=AzureMLEndpointApiType.dedicated,
    endpoint_api_key=KEY,
    content_formatter=CustomOpenAIContentFormatter(),
)

response = llm.invoke("Will humans ever solve the Collatz conjecture?")
print(response)