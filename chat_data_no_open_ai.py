import os

# for local
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

# for model
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub
from langchain.llms import HuggingFacePipeline

from langchain.indexes import VectorstoreIndexCreator

# embedding for working with my index in a vector store
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import FAISS


# for pipelining
import torch
from transformers import pipeline
from langchain import PromptTemplate, LLMChain


def get_auth():
    return os.getenv("HUGGINGFACE_HUB_API_TOKEN")


def read_document():
    loader = PdfReader("./Why_the_magic_number_seven_plus_or_minus_two.pdf")

    raw_text = ''
    for i, page in enumerate(loader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # pages = loader.load_and_split()
    return raw_text


def predict_local():

    docs = read_document()

    splitter = CharacterTextSplitter(separator="\n",
                                     chunk_size=1000,
                                     chunk_overlap=200,
                                     length_function=len)
    texts = splitter.split_text(docs)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_texts(texts, embeddings)

    # index=VectorstoreIndexCreator(embeddings=embeddings,text_splitter=splitter).from_loaders(docs)

    query = "What are the General limitations on human performance?"

    resp = db.similarity_search(query)
    print(str(resp[0].page_content))
    return resp


def predict_model():
    docs = predict_local()

    llm = HuggingFaceHub(repo_id="stabilityai/stablelm-tuned-alpha-3b",
                         model_kwargs={"temperature": 0, "max_length": 64})
    # llm = HuggingFaceHub(repo_id="databricks/dolly-v2-3b",model_kwargs={"temperature": 0, "max_length": 64})

    chain = load_qa_chain(llm, chain_type="stuff")
    query = "What are the General limitations on human performance?"
    print('---------')
    print(chain.run(input_documents=docs, question=query))


def pipelining():
    generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16,
                             trust_remote_code=True, device_map="auto", return_full_text=True)
    prompt = PromptTemplate(input_variables=[
                            "instruction", "context"], template="{instruction}\n\nInput:\n{context}")
    hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
    llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)

    docs = predict_local()
    query = "What are the General limitations on human performance?"
    print(llm_chain.predict(instruction=query, context=docs[0]).lstrip())


def main():
    pipelining()


if __name__ == "__main__":
    main()
