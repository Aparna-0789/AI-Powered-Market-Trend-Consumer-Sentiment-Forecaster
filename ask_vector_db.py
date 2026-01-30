from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# load embedding model 

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# load faiss index 
vector_db = FAISS.load_local(
    "consumer_sentiment_faiss",
    embeddings,
    allow_dangerous_deserialization = True
)

# query="Top news related to mobile accesories"
# query="common complaints in beauty care products"
query="why home appliance having good reviews"


# apply similarity search
results = vector_db.similarity_search(query, k=10)

# display result 
retrived_documents=[]
for i, r in enumerate(results,1):
    print(f"\nResult {i}")
    print("Text", r.page_content)
    print("metadata", r.metadata)
    print("="*60)
    retrived_documents.append(r.page_content)
    
prompt=f"""

    You are a market intelligence analyst
    
    using only the information from the provided context
    
    give response based on the question
    
    do not use bullet points, headings, or sections
    do not add external knowledge
    
    Context:
    {retrived_documents}
    
    Question:
    {query}
    
    Answer:
"""   
    
    
# gemini code 

from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
# The client gets the API key from the environment variable `GEMINI_API_KEY`.

# by this function i will load all evironment variables into my code 
load_dotenv()


client = genai.Client(api_key=os.getenv("Gemini_Api_key"))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
        
        temperature=0.2
    ),
)

# temparture lies between 0 to 1
print(response.text)    