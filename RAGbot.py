from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms.ollama import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
import os
import shutil
from chromadb.api.client import SharedSystemClient
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts.base import BasePromptTemplate
from scrapePDF import scrape_and_download

urls = [
    'https://wtsh.de/de/foerderprogramme',
    'https://wtsh.de/de/foerderprogramm-aussenwirtschaftsfoerderung--gemeinschaftsbuero',
    'https://wtsh.de/de/einstiegsfoerderung-fuer-innovationsvorhaben-von-kmu-eik-transfer',
    'https://wtsh.de/de/bif-modul-1-prozess-und-organisationsinnovationen',
    'https://wtsh.de/de/bif-modul-2-entwicklungsvorhaben',
    'https://wtsh.de/de/bif-modul-3-komplexe-forschungs-und-entwicklungsvorhaben',
    'https://wtsh.de/de/einstiegsfoerderung-fuer-innovationsvorhaben-von-kmu-eik-seed',
    'https://wtsh.de/de/fit-verbundvorhaben',
    'https://wtsh.de/de/foerderung-einer-ressourceneffizienten-kreislaufwirtschaft',
    'https://wtsh.de/de/foerderung-energieeinspar-energieeffizienztechnologien-energieinnovationen-ehoch3',
    'https://wtsh.de/de/foerderung-niedrigschwelliger-innovativer-digitalisierungsmassnahmen-in-kleinen-unternehmen',
    'https://wtsh.de/de/foerderung-von-digitalen-spielen',
    'https://wtsh.de/de/ladeinfrastruktur-fuer-elektrofahrzeuge-2',
    'https://wtsh.de/de/energiewende-foerderaufruf',
    'https://wtsh.de/de/aufbau-einer-nachhaltigen-wasserstoffwirtschaft---wasserstoffrichtlinie'
]

scrape_and_download(urls, "downloaded_pdfs")

embedding_function = SentenceTransformerEmbeddings(model_name="multi-qa-MiniLM-L6-dot-v1")
def load_and_split(chunk_size, chunk_overlap):
    loader = PyPDFDirectoryLoader(path="C:\\Users\\linus\\RAG_wtsh_foerderrichtlinie\\downloaded_pdfs")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = loader.load_and_split(splitter)
    return splits

def process_documents(chunk_size, chunk_overlap):
    if not os.path.exists("embeddings"):
        splits = load_and_split(chunk_size, chunk_overlap)
        db = Chroma.from_documents(collection_name="embeddings", documents=splits, embedding=embedding_function, persist_directory="embeddings")
        db.persist()
        return db
    else:
        print("Embeddings existieren bereits im Verzeichnis. Keine Aktion erforderlich.")
        return None

def re_process_documents(chunk_size, chunk_overlap):
    # Check if the database directory exists and delets it
    if os.path.exists("embeddings"):
        shutil.rmtree("embeddings")
    splits = load_and_split(chunk_size, chunk_overlap)
    db = Chroma.from_documents(collection_name="embeddings", documents=splits, embedding=embedding_function, persist_directory="embeddings")
    db.persist()
    return db

def QA_Chain (user_query, template, selected_llm_model, stop_sign):
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "input"],
                                     template=template
                                     )
    llm = Ollama(model=selected_llm_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                 stop=[stop_sign], )
    db = Chroma(collection_name="embeddings", persist_directory="embeddings", embedding_function=embedding_function)
    retriever = db.as_retriever()

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt = QA_CHAIN_PROMPT
    )
    chain = create_retrieval_chain(retriever, combine_docs_chain)

    result = chain.invoke({"input": user_query})

    # delet chroma instanc so that a revectorisation could be done https://github.com/langchain-ai/langchain/discussions/17554
    db._client._system.stop()
    SharedSystemClient._identifer_to_system.pop(db._client._identifier, None)
    return result

def app():

    st.title('WTSH Q&A')

    llm_models = ["llama3", "phi3"]

    col1, col2 = st.columns(2)
    with col1:
        selected_llm_model = st.radio("Wählen Sie ein LLM", llm_models)
    with col2:
        if selected_llm_model == "llama3":
            st.link_button("Meta Llama 3 Community License Agreement",'https://llama.meta.com/llama3/license/')
        elif selected_llm_model == "phi3":
            st.link_button("phi-3 MIT License", 'https://ollama.com/library/phi3:latest/blobs/fa8235e5b48f')
        else:
            st.write("")

    with st.sidebar:
        st.header("Einstellung")
        # Inputs for chunk size and overlap in the sidebar
        chunk_size = st.number_input("Chunk size", value=1000, min_value=100, max_value=3000, step=10)
        chunk_overlap = st.number_input("Chunk overlap", value=200, min_value=0, max_value=400, step=5)

        # Button to load and process documents in the sidebar
        if st.button('Dokumente neu vektorisieren'):
            # Assuming process_documents function is defined elsewhere and returns a Chroma object
            re_process_documents(chunk_size, chunk_overlap)
            st.success('Dokumente wurden erfolgreich vektorisiert.')

    user_query = st.text_input("Wie kann ich Ihnen weiterhelfen?")

    templatePhi3 = """"Du bist ein professioneller Berater welcher Unternehmen bei der
     Auswahl von Förderprogrammen begleitet. Du sprichst deutsch. Begrenze die Antworten auf 500 Satzzeichen.
     Nutze diese Informationen: {context}
     Frage: {input}
     Antwort:
     <|end|>"""
    templateLlama3 = """<|begin_of_text|><|start_header_id|>Benutzer<|end_header_id|>
    Du bist ein professioneller Berater welcher Unternehmen bei der
     Auswahl von Förderprogrammen begleitet. Du sprichst deutsch. Begrenze die Antworten auf maximal 500 Satzzeichen. 
     Nutze diese Informationen: {context}
     Frage: {input}<|eot_id|><|start_header_id|>Berater<|end_header_id|>
    """

    if selected_llm_model and st.button('Start Search'):

        if selected_llm_model == "llama3":
            template = templateLlama3
            stop_sign = "<|eot_id|>"
        else:
            template = templatePhi3
            stop_sign = "<|end|>"

        result = QA_Chain(user_query, template, selected_llm_model, stop_sign)

        if user_query and result:
            try:
                if result:
                        st.write(result)  # Adjust based on how your results are structured
                else:
                    st.write("No results found.")
            except Exception as e:
                st.write("An error occurred:", str(e))
        else:
            st.write("Database not loaded or query is empty.")

if __name__ == '__main__':
    app()
