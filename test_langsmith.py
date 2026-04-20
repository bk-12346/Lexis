import sys
sys.path.append('.')

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke("Say hello in one sentence.")
print(response.content)