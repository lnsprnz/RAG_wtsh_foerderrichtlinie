from langchain_community.document_loaders import PyPDFLoader

def load_document():

    loader = PyPDFLoader()
    pages = loader.load_and_split()

    return pages

print(load_document())

def split_document(pages):

    text_chunk = "Example"

    return text_chunk
def embedding_document (text_chunk):










