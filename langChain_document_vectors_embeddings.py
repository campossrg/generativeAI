import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv
load_dotenv()

openai.api_key  = os.getenv('OPENAI_API_KEY')

from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/MachineLearning-Lecture03.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)

print("Splits legnth: " + str(len(splits)))

from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

"""
sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

import numpy as np

print(np.dot(embedding1, embedding2))
print(np.dot(embedding1, embedding3))
print(np.dot(embedding2, embedding3))
"""

from langchain.vectorstores import Chroma

persist_directory = 'docs/chroma/'

#!rm -rf ./docs/chroma  # remove old database files if any

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())