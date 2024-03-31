from langchain_community.document_loaders import PyPDFLoader


def load_document():
    loader = PyPDFLoader(
        r"C:\Users\linus\OneDrive\01_Master\TAEM_WING_HSOS\KI im betrieblichen Kontext\Literatur\rili_betriebliche-innovation.pdf")
    pages = loader.load()
    for page in pages:
        print(str(page.metadata["page"]) + ":", page.page_content)


def split_document(pages):
    text_chunk = "Example"

    return text_chunk


def embedding_document(text_chunk):
    return


load_document()
