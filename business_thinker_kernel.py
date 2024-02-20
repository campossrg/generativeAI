import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion

"""
# Privative LLMs

kernel = sk.Kernel()

useAzureOpenAI = False

if useAzureOpenAI:
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    kernel.add_text_completion_service("azureopenai", AzureChatCompletion(deployment, endpoint, api_key))
else:
    #api_key, org_id = sk.openai_settings_from_dot_env()
    #kernel.add_text_completion_service("openai", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key, org_id))

print("You made a kernel!")
"""

import semantic_kernel as sk
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion

kernel = sk.Kernel()

kernel.add_text_completion_service(
    "huggingface",
    HuggingFaceTextCompletion("gpt2", task="text-generation")
)

print("You made an open source kernel using an open source AI model!")