import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
# Initialize pinecone
pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment=os.environ.get('PINECONE_API_ENV')
)

class DocAgent:
    def __init__(self, index_name=None):
        self.index_name = index_name
        self.llm = OpenAI(temperature=0)
        self.chain = load_qa_chain(self.llm, chain_type="stuff")
        self.embeddings = OpenAIEmbeddings()

    def create_index(self):
        if not self.index_name:
            raise ValueError("Index name not provided")

        loader = UnstructuredPDFLoader(
            "data/" + self.index_name + ".pdf")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(data)

        self.docsearch = PineconeVectorStore.from_texts(
            [t.page_content for t in texts], self.embeddings, index_name=self.index_name)

    def use_existing_index(self):
        if not self.index_name:
            raise ValueError("Index name not provided")

        self.docsearch = PineconeVectorStore.from_existing_index(
            index_name=self.index_name, embedding=self.embeddings)

    def query_index(self, query):
        if not hasattr(self, "docsearch"):
            raise ValueError(
                "Index not initialized. Use the create_index or use_existing_index method first.")

        docs = self.docsearch.similarity_search(query, include_metadata=True)
        return self.chain.run(input_documents=docs, question=query)

    def ask(self, query):
        response = self.query_index(query)
        return response



doc_agent = DocAgent(index_name="2223programaciogeneralm09")
# doc_agent.create_index()
doc_agent.use_existing_index()
question = "Pot aprovar un alumne que no vingui a classe?"
response = doc_agent.ask(question)
print(response)