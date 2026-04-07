from dotenv import load_dotenv
from langchain_community .document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

data = PyPDFLoader("document_loaders/deep-learning.pdf")
docs = data.load()




template = ChatPromptTemplate.from_messages(
    [("system", "you are a AI that summarizes the text"),
    ("human","{data}")]
)

model = ChatMistralAI( model = 'mistral-small-2506')

