from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import application_properties

chat = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", api_key=application_properties.api_key)

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)
prompt_template.messages[0].prompt