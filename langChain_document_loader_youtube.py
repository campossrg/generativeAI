import properties

import os
import openai
import sys

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

from dotenv import load_dotenv
load_dotenv()

print("API_KEY CHECK")
print("*************")
for name, value in os.environ.items():
    print(f"{name}: {value}")

api_key = os.environ.get('OPENAI_API_KEY')

if api_key is None:
    print("OPENAI_API_KEY is not set.")
else:
    print("OPENAI_API_KEY is set.")

sys.path.append('../..')
openai.api_key = os.environ['OPENAI_API_KEY']

#url="https://www.youtube.com/watch"
save_dir="docs/youtube/"
loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),
    OpenAIWhisperParser()
)
docs = loader.load()
docs[0].page_content[0:500]