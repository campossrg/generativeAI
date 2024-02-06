import os
def api_key_check():
    print("API_KEY CHECK")
    print("*************")
    for name, value in os.environ.items():
        print(f"{name}: {value}")

    api_key = os.getenv('OPENAI_API_KEY')

    if api_key is None:
        print("OPENAI_API_KEY is not set.")
    else:
        print("OPENAI_API_KEY is set.")