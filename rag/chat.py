from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model,
)

user_query = input("Enter your query: ")

search_results = vector_db.similarity_search(query=user_query)

context = "\n\n\n".join(f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}\n"
    for result in search_results)

SYSTEM_PROMPT = f"""
You are a helpful assistant that answers user's query based on the available context retrieved from a pdf along with page_contents and page_number. 
You should only answer the user based on the following context and navigate the user  to open the right page number to know more.

Context:
{context}
"""

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

response = client.chat.completions.create(
    model="gemma4:e2b",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ],
)

print(response.choices[0].message.content)
