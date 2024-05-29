from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint, CustomOpenAIChatContentFormatter
from langchain_community.llms.azureml_endpoint import AzureMLEndpointApiType
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

URI = "https://repro-toxicity-xrbuc.eastus2.inference.ml.azure.com/score"
KEY = "9aIh4YiafyykA1e9okBCiwz6gupwDskX"

SYSTEM_MESSAGE_PROMPT = """ You are a chat bot that answers questions about test guidelines."""

chroma_path = "../chroma"

class ChatBot:
    """
    Input:
        pdf_path (str): Path to the PDF folder that you want to use for your queries
    """

    def __init__(self, path: str = '') -> None:
        vectordb = Chroma(persist_directory=chroma_path, embedding_function = OllamaEmbeddings(model='nomic-embed-text'))
        retriever = vectordb.as_retriever()
        
        self.llm = AzureMLChatOnlineEndpoint(
            endpoint_url=URI,
            endpoint_api_type=AzureMLEndpointApiType.dedicated,
            endpoint_api_key=KEY,
            content_formatter=CustomOpenAIChatContentFormatter(),
            model_kwargs={"max_tokens": 512},
        )

        SYSTEM_TEMPLATE = """
        Answer the user's questions based on the below context. 
        If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

        <context>
        {context}
        </context>
        """

        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_TEMPLATE,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        document_chain = create_stuff_documents_chain(self.llm, question_answering_prompt)

        query_transform_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="messages"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
                ),
            ]
        )
        
        query_transforming_retriever_chain = RunnableBranch(
            (
                lambda x: len(x.get("messages", [])) == 1,
                # If only one message, then we just pass that message's content to retriever
                (lambda x: x["messages"][-1].content) | retriever,
            ),
            # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
            query_transform_prompt | self.llm | StrOutputParser() | retriever,
        ).with_config(run_name="chat_retriever_chain")

        self.conversational_retrieval_chain = RunnablePassthrough.assign(
            context=query_transforming_retriever_chain,
        ).assign(
            answer=document_chain,
        )

    def query(self, message, chat_history):
        chat_history.append(HumanMessage(content=message))
        response = self.conversational_retrieval_chain.invoke({
            "messages": [
                HumanMessage(content=message),                
            ]
        })
        return response['answer']