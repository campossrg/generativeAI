import properties
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
#from IPython.display import display, Markdown

file = 'OutdoorClothingCatalog_1000.csv'

import csv

print("Starting...")

def read_csv():
    print("reading the file")
    with open(file) as f:
        reader = csv.reader('C:\documents\courses\generative_ai\generativeAI\OutdoorClothingCatalog_1000.csv')
        return [row for row in reader]
    
print(read_csv)