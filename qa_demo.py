from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores import Cassandra
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import chainlit as cl
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASTRADB_ID = os.getenv('ASTRADB_ID')
ASTRADB_SECRET = os.getenv('ASTRADB_SECRET')


'''
    Cassandra AstraDB part
'''
keyspace_name = "qa_docs"
table_name = "qa_table"

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

cloud_config= {
  'secure_connect_bundle': '/content/sample_data/secure-connect-chatgpt.zip'
}
auth_provider = PlainTextAuthProvider(ASTRADB_ID, ASTRADB_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
#session = cluster.connect()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


@cl.langchain_factory(use_async=True)
async def init():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!", accept=["text/plain"]
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Decode the file
    loader = TextLoader(file.path)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)


    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # create the vectorestore to use as the index
    session = cluster.connect()

    db = Cassandra.from_documents(
        documents=texts,
        embedding=embeddings,
        session=session,
        keyspace=keyspace_name,
        table_name=table_name,
        metadatas=metadatas)


    # Create a chain that uses the Chroma vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        #ChatOpenAI(temperature=0),
        llm=OpenAI(openai_api_key = OPENAI_API_KEY),
        chain_type="stuff",
        retriever = db.as_retriever(),
        return_source_documents=True
    )

    # Save the metadata and texts in the user session
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)

    # Let the user know that the system is ready
    await msg.update(content=f"`{file.name}` processed. You can now ask questions!")


    return chain


@cl.langchain_postprocess
async def process_response(res):
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=source_elements).send()
