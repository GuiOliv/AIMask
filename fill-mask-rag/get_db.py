from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from promptflow import tool


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool():
    dataset_name = "BotatoFontys/DataBank"

    loader = HuggingFaceDatasetLoader(dataset_name)

    # Load the data
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

# 'data' holds the text you want to split, split the text into documents using the text splitter.
    docs = text_splitter.split_documents(data)

    modelPath = "sentence-transformers/all-MiniLM-L6-v2"

    model_kwargs = {'device':'cuda'}

    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    return FAISS.from_documents(docs, embeddings)