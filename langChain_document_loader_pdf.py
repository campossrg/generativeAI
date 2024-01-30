import properties

import os
import openai
import sys
#from langchain.document_loaders import PyPDFLoader  ## deprecated
from langchain_community.document_loaders import PyPDFLoader

sys.path.append('../..')
openai.api_key = properties.api_key

loader = PyPDFLoader("docs/ARVA.pdf")
pages = loader.load()

print(len(pages))
page = pages[0]
print(page.page_content[0:500])
print(page.metadata)