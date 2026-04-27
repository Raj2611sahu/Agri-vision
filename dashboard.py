import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from process_plant_image import process_plant_image, refine_with_knowledge

# 1. Initialize Embeddings (Runs locally on your CPU)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



def build_knowledge_base( folder_path = "./knowledge_base" ):
    documents = []
    for file in os.listdir( folder_path ):
        if file.endswith( ".pdf" ):
            loader = PyPDFLoader( os.path.join( folder_path, file ) )
            documents.extend( loader.load() )
    
    # Split text into chunks Gemma 4 can digest easily
    text_splitter = RecursiveCharacterTextSplitter( chunk_size = 1000, chunk_overlap = 100 )
    chunks = text_splitter.split_documents( documents )
    
    # Create a local vector store
    vectorstore_data = Chroma.from_documents( documents = chunks, embedding = embeddings, persist_directory = "./db" )
    return vectorstore_data


@st.cache_resource
def load_vectorstore():
    return build_knowledge_base("./knowledge_base")


st.title("Agri-Vision: Crop Diagnostic")
st.write("Upload a leaf photo to diagnose health issues.")

vectorstore = load_vectorstore()

def get_relevant_context( query, vectorstore_data ):
    docs = vectorstore_data.similarity_search( query, k = 3 )
    return "\n".join( [doc.page_content for doc in docs] )

uploaded_file = st.file_uploader(
    "Choose a plant image...",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Crop Photo", use_container_width=True)

    with st.spinner("Analyzing visual symptoms..."):
        vision_diagnosis = process_plant_image(uploaded_file)

    st.subheader("Visual Diagnosis")
    st.write(vision_diagnosis)

    with st.spinner("Consulting agricultural knowledge base..."):
        context = get_relevant_context( vision_diagnosis,
            vectorstore )
        final_diagnosis = refine_with_knowledge(
            context,
            vision_diagnosis
        )

    st.subheader("Botanist Analysis (Knowledge‑Grounded)")
    st.write(final_diagnosis)