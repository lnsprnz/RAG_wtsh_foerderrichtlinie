from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
import os
import shutil
from scrapePDF import scrape_and_download
import time
import logging

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
templatePhi3 = """" 
***Instruction***
Persona: Professioneller Förderberater
Sprache: deutsch
Länge der Antworten: 150 Wörter
 Beantworte diese Frage: {input} 
 Mit diesen Informationen:{context}
 Antwort:
 <|end|>"""
templateLlama3 = """<|begin_of_text|><|start_header_id|>Benutzer<|end_header_id|>
Du bist ein professioneller Berater welcher Unternehmen bei der
 Auswahl von Förderprogrammen begleitet. Du sprichst deutsch. Begrenze die Antworten auf maximal 100 Satzzeichen. 
 Nutze diese Informationen: {context}
 Frage: {input}<|eot_id|><|start_header_id|>Berater<|end_header_id|>
"""

embedding_function = SentenceTransformerEmbeddings(model_name="LLukas22/all-MiniLM-L12-v2-embedding-all")
def load_and_split(chunk_size, chunk_overlap):
    loader = PyPDFDirectoryLoader(path="C:\\Users\\linus\\RAG_wtsh_foerderrichtlinie\\downloaded_pdfs")
    splitter = RecursiveCharacterTextSplitter(separators= ["\n","."],  chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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

def build_retriever(db, k, fetch_k,  lambda_mult):
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k,"lambda_mult": lambda_mult})
    return retriever


def build_QA_Chain (template, selected_llm_model, stop_sign, retriever):
    # Define Prompt Template
    Prompt_Template = PromptTemplate(input_variables=["context", "input"],template=template )
    #Define llm
    llm = Ollama(model=selected_llm_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),stop=[stop_sign],)

    #Build Chain
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt = Prompt_Template
    )
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    return chain
def invoke_QA_Chain(chain, user_query):
    result = chain.invoke({"input": user_query})
    return result

def close_db (db):
    # delet chroma instanc so that a revectorisation could be done https://github.com/langchain-ai/langchain/discussions/17554
    db._client._system.stop()
    return None

def setup_logging():
    # Configure logging
    logging.basicConfig(filename='app.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def app():

    setup_logging()
    with st.sidebar:
        llm_models = ["llama3", "phi3"]
        st.header("Einstellung")
        st.subheader("Dokumente")
        # Inputs for chunk size and overlap in the sidebar
        chunk_size = st.slider("Chunk size", value=150, min_value=50, max_value=250, step=10)
        chunk_overlap = st.slider("Chunk overlap", value=0, min_value=0, max_value=30, step=5)


        if st.button('Dokumente aktualisieren'):
            # Assuming process_documents function is defined elsewhere and returns a Chroma object
            scrape_and_download(urls, "downloaded_pdfs")
            re_process_documents(chunk_size, chunk_overlap)
            st.success('Dokumente wurden erfolgreich vektorisiert.')

        st.divider()

        st.subheader("Retriever")
        #Change the dynamics of the mmr retriever
        k = st.slider ("k", value=4, min_value=1, max_value=40, step=1)
        fetch_k = st.slider ("fetch_k",  value=20, min_value=20, max_value=350, step=1)
        lambda_mult = st.slider ("lambda_mult", value=0.5, min_value=0.0, max_value=1.0, step=0.01)

        st.divider()

        st.subheader("LLM")
        #select a LLM
        selected_llm_model = st.radio("LLM Model", llm_models)
        if selected_llm_model == "llama3":
            st.markdown("[Meta Llama 3 Community License Agreement](https://llama.meta.com/llama3/license/)")
        elif selected_llm_model == "phi3":
            st.markdown("[phi-3 MIT License](https://ollama.com/library/phi3:latest/blobs/fa8235e5b48f)")

    st.title('WTSH Q&A')

    user_query = st.text_input("Wie kann ich Ihnen weiterhelfen?")

    if selected_llm_model and st.button('Start Search'):
        start_time = time.time()
        if selected_llm_model == "llama3":
            template = templateLlama3
            stop_sign = "<|eot_id|>"
        else:
            template = templatePhi3
            stop_sign = "<|end|>"

        db = Chroma(collection_name="embeddings", persist_directory="embeddings", embedding_function=embedding_function)
        retriever = build_retriever(db, k, fetch_k, lambda_mult)
        chain = build_QA_Chain(template, selected_llm_model, stop_sign, retriever)
        result = invoke_QA_Chain(chain, user_query)
        time_taken = time.time() - start_time
        close_db(db)

        if user_query and result:
            try:
                if result:
                        st.write(result)  # Adjust based on how your results are structured
                        logging.info(
                            f"Query: {user_query}, Result: {result}, Time: {time_taken}s, Parameters: Chunk Size: {chunk_size}, Chunk Overlap: {chunk_overlap}, k: {k}, fetch_k: {fetch_k}, lambda_mult: {lambda_mult}, LLM: {selected_llm_model}")
                else:
                    st.write("No results found.")
            except Exception as e:
                st.write("An error occurred:", str(e))
        else:
            st.write("Database not loaded or query is empty.")

if __name__ == '__main__':
    app()
