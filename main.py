__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button
# from dotenv import load_dotenv
# load_dotenv()

# 제목
st.title("ChatPDF")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=["pdf"])
st.write("---")

# OpenAI 키 입력받기
openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

# Buy me a coffee
button(username="longbinjin", floating=True, width=221)

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    st.write(f"Temporary file path: {temp_filepath}")
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_documents(pages)
    # print(texts[2])

    # Embeddings
    embeddings_model = OpenAIEmbeddings(
        model='text-embedding-3-large',
        openai_api_key=openai_key,
        # dimensions=1024,
    )

    import chromadb
    chromadb.api.client.SharedSystemClient.clear_system_cache()


    # Chroma DB
    db = Chroma.from_documents(texts, embeddings_model)

    # User Input
    st.header("PDF에게 질문해보세요!")
    question = st.text_input("질문을 입력해주세요!")

    if st.button("질문하기"):
        with st.spinner("Wait for it..."):
            # Retriever
            llm = ChatOpenAI(model='gpt-5-nano', temperature=0)
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=db.as_retriever(),
                llm=llm,
            )

            # Prompt Template
            prompt = hub.pull("rlm/rag-prompt")

            # Generate
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            rag_chain = (
                {"context": retriever_from_llm | format_docs,
                "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # Question
            result = rag_chain.invoke(question)
            st.write(result)
