import properties
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# CSV reader
df = pd.read_csv('Data.csv')
print(df.head())

llm = ChatOpenAI(temperature=0.9, model=properties.llm_model, api_key=properties.api_key)

prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

chain = LLMChain(llm=llm, prompt=prompt)

product = "Queen Size Sheet Set"
print(chain.run(product))
