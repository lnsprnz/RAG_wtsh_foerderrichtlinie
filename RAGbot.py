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
from llama3.llama.generation import Llama

embedding_function = SentenceTransformerEmbeddings(model_name="multi-qa-MiniLM-L6-dot-v1")

def get_embedding(query):
    model = SentenceTransformerEmbeddings(model_name='multi-qa-MiniLM-L6-dot-v1')
    embedding = model.embed_query(query)
    return embedding

def load_and_split(chunk_size, chunk_overlap):
    loader = PyPDFDirectoryLoader(path="C:\\Users\\linus\\RAG_wtsh_foerderrichtlinie\\downloaded_pdfs")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = loader.load_and_split(splitter)
    return splits

def process_documents(chunk_size, chunk_overlap):
    splits = load_and_split(chunk_size, chunk_overlap)
    db = Chroma.from_documents(collection_name="embeddings", documents=splits, embedding=embedding_function, persist_directory="embeddings")
    db.persist()
    return db

def app():
    st.title('WTSH Förderchat 💰📈')
    # Create two columns
    col1, col2 = st.columns(2)
    with col1:
        st.write('Built with Meta Llama 3')
    with col2:
        st.link_button("Meta Llama 3 Community License Agreement",'https://llama.meta.com/llama3/license/')

    with st.sidebar:
        st.header("Settings")
        # Inputs for chunk size and overlap in the sidebar
        chunk_size = st.number_input("Chunk size", value=200, min_value=100, max_value=1000, step=10)
        chunk_overlap = st.number_input("Chunk overlap", value=20, min_value=0, max_value=200, step=5)

        # Button to load and process documents in the sidebar
        if st.button('Load and Process Documents'):
            # Assuming process_documents function is defined elsewhere and returns a Chroma object
            process_documents(chunk_size, chunk_overlap)
            st.success('Documents have been loaded and processed.')

    user_query = st.text_input("Enter your search query:")

    template = """"<|begin_of_text|><|start_header_id|>user<|end_header_id|>Du bist ein professioneller Berater welcher Unternehmen bei der
     Auswahl von Förderprogrammen begleitet. Du sprichst deutsch. Begrenze die Antworten auf 240 Satzeichen. 
     Hier sind relevante Informationen: {context}
     Frage: {question}
     <|eot_id|> 
     <|start_header_id|>assistant<|end_header_id|>"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)
    llm = Ollama(model="llama3",callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),stop=["<|eot_id|>"],)
    db = Chroma(collection_name="embeddings", persist_directory="embeddings", embedding_function=embedding_function)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    if st.button('Start Search'):

        result = qa_chain({"query": user_query})
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
