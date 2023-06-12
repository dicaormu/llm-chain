import os

from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain


video = "https://www.youtube.com/watch?v=NXcgOIfz71U"


def get_auth():
    return os.getenv("OPEN_AI_KEY")


def read_video():
    loader = YoutubeLoader.from_youtube_url(video, add_video_info=True)
    return loader.load()


def predict_local():
    video = read_video()
    print(type(video))
    print(video)
    llm = OpenAI(temperature=0, openai_api_key=get_auth())

    chain = load_summarize_chain(llm, chain_type="stuff", verbose=False)

    print(chain.run(video))
    # summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=video)


def main():
    predict_local()


if __name__ == "__main__":
    main()
