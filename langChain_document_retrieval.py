import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv
load_dotenv()

openai.api_key  = os.getenv('OPENAI_API_KEY')

from langchain_community.document_loaders import PyPDFLoader
