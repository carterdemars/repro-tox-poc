from difflib import HtmlDiff
from fpdf import FPDF
# from langchain.document_loaders import (
#     PyPDFLoader
# )

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from typing import Dict

from langchain_core.runnables import RunnablePassthrough

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

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

        document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

        retrieval_chain = RunnablePassthrough.assign(
            context=parse_retriever_input | retriever,
        ).assign(
            answer=document_chain,
        )

        query_transform_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="messages"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
                ),
            ]
        )

        query_transformation_chain = query_transform_prompt | llm

        query_transformation_chain.invoke(
            {
                "messages": [
                    HumanMessage(content="Can LangSmith help test my LLM applications?"),
                    AIMessage(
                        content="Yes, LangSmith can help test and evaluate your LLM applications. It allows you to quickly edit examples and add them to datasets to expand the surface area of your evaluation sets or to fine-tune a model for improved quality or reduced costs. Additionally, LangSmith can be used to monitor your application, log all traces, visualize latency and token usage statistics, and troubleshoot specific issues as they arise."
                    ),
                    HumanMessage(content="Tell me more!"),
                ],
            }
        )

        query_transforming_retriever_chain = RunnableBranch(
            (
                lambda x: len(x.get("messages", [])) == 1,
                # If only one message, then we just pass that message's content to retriever
                (lambda x: x["messages"][-1].content) | retriever,
            ),
            # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
            query_transform_prompt | llm | StrOutputParser() | retriever,
        ).with_config(run_name="chat_retriever_chain")

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

        document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

        conversational_retrieval_chain = RunnablePassthrough.assign(
            context=query_transforming_retriever_chain,
        ).assign(
            answer=document_chain,
        )

    def parse_retriever_input(params: Dict):
        return params["messages"][-1].content

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