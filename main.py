from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.gcs_directory import GCSDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma

class DocumentSplit:
    def __init__(self, content, metadata={}):
        self.page_content = content
        self.metadata = metadata

def load_Google_Cloud():
    loader = GCSDirectoryLoader(project_name="prinzmicroservice", bucket= "prinz-data")
    pages = loader.load()
    return pages
def split_document(pages):
    splitterType = RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=20)
    splits = splitterType.split_text(pages)
    return splits

def embedding_splits(splits):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(splits, embedding_function)
    return db

def process_documents():
    documents = load_Google_Cloud()
    all_splits = []
    for document in documents:
        page_content = document.page_content
        document_splits = split_document(page_content)
        all_splits.extend([DocumentSplit(split) for split in document_splits])

    db = embedding_splits(all_splits)
    return db


db = process_documents()

# query it
query = "Wie hoch sind die maximalen zuwendungsfähigen Ausgaben beim Seed-Bonus?"
result = db.similarity_search(query)

# print results
print(result[0].page_content)
