# -*- coding: utf-8 -*-
"""Demo1.3- LangChain_QA_Panel_App.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Jk9O3n8lj3_MuqWLCukzi_x71IoGPHM5

# LangChain QA Panel App

This notebook shows how to make this app:
"""


import os
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import Cassandra
import panel as pn
import tempfile

keyspace_name = "vector"
table_name = "qa_table"

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASTRADB_ID = os.getenv('ASTRADB_ID')
ASTRADB_SECRET = os.getenv('ASTRADB_SECRET')

cloud_config= {
  'secure_connect_bundle': 'secure-connect-demo-llm.zip'
}
auth_provider = PlainTextAuthProvider(ASTRADB_ID, ASTRADB_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

#from google.colab import drive
#drive.mount('/content/drive')

pn.extension('texteditor', template="bootstrap", sizing_mode='stretch_width')
pn.state.template.param.update(
    main_max_width="690px",
    header_background="#F08080",
)

#User Interface
file_input = pn.widgets.FileInput(width=300)

prompt = pn.widgets.TextEditor(
    value="", placeholder="Enter your questions here...", height=160, toolbar=False
)
run_button = pn.widgets.Button(name="Run!")

select_k = pn.widgets.IntSlider(
    name="Number of relevant chunks", start=1, end=5, step=1, value=2
)
select_chain_type = pn.widgets.RadioButtonGroup(
    name='Chain type',
    options=['stuff', 'map_reduce', "refine", "map_rerank"]
)

widgets = pn.Row(
    pn.Column(prompt, run_button, margin=5),
    pn.Card(
        "Chain type:",
        pn.Column(select_chain_type, select_k),
        title="Advanced settings", margin=10
    ), width=600
)

def qa(file, query, chain_type, k):
    # load document
    #loader = PyPDFLoader(file)
    session = cluster.connect()
    loader = TextLoader(file)
    documents = loader.load()
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    # create the vectorestore to use as the index
    db = Cassandra.from_documents(
        documents=texts,
        embedding=embeddings,
        session=session,
        keyspace=keyspace_name,
        table_name=table_name,)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_kwargs={"k": k})
    # create a chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key = OPENAI_API_KEY), chain_type=chain_type, retriever=retriever, return_source_documents=True)
    result = qa({"query": query})
    print(result['result'])
    return result

#result = qa("example.pdf", " ")

convos = []  # store all panel objects in a list

def qa_result(_):

    # save pdf file to a temp file
    if file_input.value is not None:
        file_input.save("temp.txt")

        prompt_text = prompt.value
        if prompt_text:
            result = qa(file="temp.txt", query=prompt_text, chain_type=select_chain_type.value, k=select_k.value)
            convos.extend([
                pn.Row(
                    pn.panel("", width=10),
                    prompt_text,
                    width=800
                )
            ])
            #return convos
    return pn.Column(*convos, margin=15, width=575, min_height=400)

qa_interactive = pn.panel(
    pn.bind(qa_result, run_button),
    loading_indicator=True,
)

output = pn.WidgetBox('*Output will show up here:*', qa_interactive, width=630, scroll=True)

# layout
pn.Column(
    pn.pane.Markdown("""
    ## Question Answering with your PDF file

    1) Upload a PDF. 2) Enter OpenAI API key. This costs $. Set up billing at [OpenAI](https://platform.openai.com/account). 3) Type a question and click "Run".

    """),
    pn.Row(file_input,OPENAI_API_KEY),
    output,
    widgets

).servable()

# panel serve demo1.ipynb