import streamlit as st
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from langchain_community.llms import Replicate
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import requests
import os
import re
from pinecone import Pinecone, ServerlessSpec


st.warning("""
    This is for testing purposes and may contain errors from chat responses.
""")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]
os.environ['REPLICATE_API_TOKEN'] = st.secrets["REPLICATE_API_TOKEN"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]


pc = Pinecone(api_key = st.secrets["PINECONE_API_KEY"])

# Now do stuff
if 'myfirstpineconeindex' not in pc.list_indexes().names():
    pc.create_index(
        name='myfirstpineconeindex',
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")

url_input = st.sidebar.text_input("Enter an url of PDF file")

def download_file(url):
    try:
        response = requests.get(url)
        with open(get_filepath(url), 'wb') as f:
            f.write(response.content)
        return True
    except:
        return False

def extractuuid(url):
    return re.findall(r"https://ds7dev.shareok.org/bitstreams/([A-Za-z0-9\-]+)", url)[0]

def get_filename(url):
    return extractuuid(url) + '.pdf'

def get_filepath(url):
    return os.path.join("pages/files/"+get_filename(url))

def get_documents(filepath):
    loader = PyMuPDFLoader(filepath)
    documents = loader.load()
    st.write("documents: ")
    st.write(documents)
    return documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

embeddings = HuggingFaceEmbeddings()
index_name = "myfirstpineconeindex"
index = pc.Index(index_name)

# Initialize Replicate Llama2 Model
llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    model_kwargs={"temperature": 0.75, "max_length": 3000}
)

def get_texts(filepath):
    return text_splitter.split_documents(get_documents(filepath))


if url_input:
    if download_file(url_input):
        try:
            vectordb = PineconeVectorStore.from_documents(get_texts(get_filepath(url_input)), embeddings, index_name=index_name)
        except Exception as inst:
            st.write("Can't create db");
            st.write(inst);
    else:
        st.write("Failed to download bitstream")

user_input = st.chat_input("Enter your message")

if user_input:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectordb.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True
    )

    chat_pdf_history = []
    query = user_input
    result = qa_chain.invoke({'question': query, 'chat_history': chat_pdf_history})
    st.write('Answer: ' + result['answer'] + '\n')
    chat_pdf_history.append((query, result['answer']))



# pc.delete_index("quickstart")