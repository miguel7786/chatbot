import os
from langchain_openai.embeddings import OpenAIEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ensure the `data` directory exists
if not os.path.exists("data"):
    os.makedirs("data")
    print("The 'data' directory was missing and has been created. Please add your .txt files to this folder.")
    exit()

# Load the documents
documents = []
for filename in os.listdir("data"):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join("data", filename))
        documents.extend(loader.load())

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# Initialize the embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))  # Use environment variable for safety
vectorstore = FAISS.from_documents(documents, embeddings)

# Save the vector store
if not os.path.exists("vectorstore"):
    os.makedirs("vectorstore")
vectorstore.save_local("vectorstore")

print("Vector store built and saved successfully.")
