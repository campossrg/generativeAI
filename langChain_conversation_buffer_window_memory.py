import properties
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=1)   # This stores only one message at a time

memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})

memory.load_memory_variables({})

llm = ChatOpenAI(temperature=0.0, model=properties.llm_model, api_key=properties.api_key)
memory = ConversationBufferWindowMemory(k=3)  # This stores until 3 messages at a time and then he can remember Adrew name :)
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=False
)

print(conversation.predict(input="Hi, my name is Andrew"))
print(conversation.predict(input="What is 1+1?"))
print(conversation.predict(input="What is my name?"))