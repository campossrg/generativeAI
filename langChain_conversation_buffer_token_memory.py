import properties
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory

llm = ChatOpenAI(temperature=0.0, model=properties.llm_model, api_key=properties.api_key)
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)  # This stores a maximum of 50 tokens

memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})

print(memory.load_memory_variables({}))