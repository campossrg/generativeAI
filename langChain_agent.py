import properties

#from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
#from langchain.tools.python.tool import PythonREPLTool
#from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI
import wikipedia

print("Starting...")
print(wikipedia.__file__)

llm = ChatOpenAI(temperature=0, model=properties.llm_model, api_key=properties.api_key)

tools = load_tools(["llm-math","wikipedia"], llm=llm)

agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
result = agent(question) 