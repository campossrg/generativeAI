import properties

import os
import openai

from langchain.document_loaders import WebBaseLoader
from dotenv import load_dotenv
load_dotenv()

#openai.api_key = os.environ['OPENAI_API_KEY']

#loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/37signals-is-you.md")
loader = WebBaseLoader("https://www.google.es/search?q=earth&sca_esv=729edc120b86c4d9&sxsrf=ACQVn09j0pHbvIDr0QEq0PF6xCrB_8tMSQ%3A1706604106346&source=hp&ei=Sra4ZYbnEeHukdUPqIWvmAY&iflsig=ANes7DEAAAAAZbjEWi3KPeguZphrDg1vVF48Fx_4Owd7&ved=0ahUKEwiGtJzc24SEAxVhd6QEHajCC2MQ4dUDCA8&uact=5&oq=earth&gs_lp=Egdnd3Mtd2l6IgVlYXJ0aDILEC4YgwEYsQMYgAQyCxAAGIAEGLEDGIMBMgsQABiABBixAxiDATILEC4YgAQYsQMYgwEyCxAAGIAEGLEDGIMBMggQLhiABBixAzIOEC4YgAQYsQMYxwEY0QMyBRAAGIAEMgUQLhiABDIOEC4YgAQYsQMYgwEY1AJI2CJQ_RpY5CBwAXgAkAEAmAFroAHyA6oBAzIuM7gBA8gBAPgBAagCCsICBxAjGOoCGCfCAgQQIxgnwgIREC4YgAQYsQMYgwEYxwEY0QPCAg4QLhjHARixAxjRAxiABMICCBAuGLEDGIAEwgIREC4YgAQYsQMYgwEYxwEYrwHCAggQABiABBixA8ICCxAuGIAEGMcBGK8B&sclient=gws-wiz")

docs = loader.load()

print(docs[0].page_content[:500])