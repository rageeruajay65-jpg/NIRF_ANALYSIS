from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    max_retries=3
)

try:
    response = llm.invoke("Reply: Gemini is working!")
    print(response.content)

except Exception as error:
    print("Gemini connection failed:")
    print(error)

    