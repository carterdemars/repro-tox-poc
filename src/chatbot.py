from difflib import HtmlDiff
from fpdf import FPDF
# from langchain.document_loaders import (
#     PyPDFLoader
# )

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import Chroma

from langchain.agents import Tool, OpenAIFunctionsAgent, AgentExecutor
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.embeddings import GPT4AllEmbeddings
#from langchain.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
    ReduceDocumentsChain,
    MapReduceDocumentsChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.memory import ConversationBufferMemory
from typing import List

from langchain_community.llms.azureml_endpoint import (
    AzureMLEndpointApiType,
    AzureMLOnlineEndpoint,
    CustomOpenAIContentFormatter,
)

from langchain_community.llms.azureml_endpoint import (
    AzureMLEndpointApiType
)
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint, CustomOpenAIChatContentFormatter

URI = 'https://repro-toxicity-mvdiw.eastus2.inference.ml.azure.com/score'
KEY = 'jAYo2rLdFte9ZF35LanyBXHE9YUW072P'
MEMORY_KEY = "memory"

SYSTEM_MESSAGE_PROMPT = """ You are a chat bot that answers questions about test guidelines. 
"""

class ChatBot:
    """
    Input:
        pdf_path (str): Path to the PDF folder that you want to use for your queries
    """
    def __init__(self, path: str = '') -> None:
        # self.full_document = PyPDFLoader(pdf_path).load()
        # split_documents = RecursiveCharacterTextSplitter(
        #     chunk_size=500, chunk_overlap=0
        # ).split_documents(self.full_document)

        #self.load_documents()

        # self.tools = [
        #     Tool.from_function(
        #         name="document_QA_tool",
        #         func=self.qa_retrival_bot.query,
        #         description="""
        #             Used when you need to answer a general question about the document's contents. This is useful for
        #             when the user is asking questions about the document, and isn't asking for you to summarize the
        #             document.
        #             Input:
        #                 general_question (str): The user's general question concerning the document's contents
        #         """,
        #     )
        # ]

        self.llm = AzureMLChatOnlineEndpoint(
            endpoint_url=URI,
            endpoint_api_type=AzureMLEndpointApiType.dedicated,
            endpoint_api_key=KEY,
            content_formatter=CustomOpenAIChatContentFormatter(),
        )

        # self.prompt = OpenAIFunctionsAgent.create_prompt(
        #     system_message=SystemMessage(content=SYSTEM_MESSAGE_PROMPT),
        #     extra_prompt_messages=[
        #         MessagesPlaceholder(variable_name=MEMORY_KEY),
        #     ],
        # )

        self.memory = ConversationBufferMemory(
            memory_key=MEMORY_KEY,
            return_messages=True,
            input_key="input",
            output_key="output",
        )

        # self.qa_chat = ConversationalRetrievalChain.from_llm(
        #     self.llm, retriever=self.vectorstore.as_retriever(), memory=self.memory
        # )

        # self.agent = OpenAIFunctionsAgent(
        #     llm=self.llm, tools=self.tools, prompt=self.prompt
        # )

        # self.agent_executor = AgentExecutor(
        #     agent=self.agent,
        #     tools=self.tools,
        #     memory=self.memory,
        #     verbose=True,
        #     return_intermediate_steps=True,
        # )

    def load_documents(self):

        # USE BING'S CHUNKING ALGORITHM

        split_documents = []
        
        self.vectorstore = Chroma.from_documents(
            documents=split_documents, embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        )

    def query(self, message, chat_history, message_placeholder):
        """
        """
        # output = self.agent_executor(
        #     f"Use the document to answer the following question: {prompt}"
        # )
        response = self.llm.invoke(message)
        return response
        

# class QARetrievalBot:
#     """
#     Bot that handles general questions about the contents of a PDF.
#     Does so using Langchain's ConversationalRetrievalChain, which stores
#     different document parts as embeddings in a vectore store, and performs
#     a similarity search between these embeddings and the user's prompt.
#     See here for more https://python.langchain.com/docs/use_cases/question_answering/

#     Input:
#         split_documents (List[Document]): List of document split into its different
#             parts
#     """
#     def __init__(self, split_documents: List[Document]) -> None:
#         self.memory = ConversationBufferMemory(
#             memory_key="chat_history", return_messages=True
#         )
#         self.vectorstore = Chroma.from_documents(
#             documents=split_documents, embedding=GPT4AllEmbeddings()
#         )
#         self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
#         self.qa_chat = ConversationalRetrievalChain.from_llm(
#             self.llm, retriever=self.vectorstore.as_retriever(), memory=self.memory
#         )

#     def query(self, general_question: str) -> str:
#         return self.qa_chat({"question": general_question})