from openai import OpenAI


client = OpenAI(
    api_key='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'
)

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."}
  ]
)

print(response)


