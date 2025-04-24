import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

### RAG DATA ###
df = pd.read_csv('realdonaldtrump.csv')
df = df.head(50)  # Limit to 1000 tweets for testing
docs = [Document(page_content=tweet) for tweet in df['content']]
embeddings = OpenAIEmbeddings(
    model="BAAI-bge-large-en-v1-5",
    openai_api_key="sk-lTDIFvqqhJRPJWah-C5FWA",
    openai_api_base="https://chatapi.akash.network/api/v1",
)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()


### RUNNIGN LLM PART ###
system_prompt = """
You are a tweet imitator bot. When given a query to generate a tweet to reply or a new tweet,
try to use retrieved similar tweets to help augment your answer.
Keep the answer concise.

Retrieved similar tweets: {context}:"""

question = """Generate a tweet to reply to this: What is the best way to learn Python?"""
docs = retriever.invoke(question)
docs_text = "".join(d.page_content for d in docs)
system_prompt_fmt = system_prompt.format(context=docs_text)
print("System Prompt:")
print(system_prompt_fmt)

model = ChatOpenAI(
    model="Meta-Llama-3-1-8B-Instruct-FP8",
    openai_api_key="sk-lTDIFvqqhJRPJWah-C5FWA",
    openai_api_base="https://chatapi.akash.network/api/v1",
)

response = model.invoke([SystemMessage(content=system_prompt_fmt),
                         HumanMessage(content=question)])

print(response.content)