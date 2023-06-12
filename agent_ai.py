import os

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI


def get_auth():
    return os.getenv("OPEN_AI_KEY")


def agent_prediction():
    llm = OpenAI(temperature=0, openai_api_key=get_auth())
    tools = load_tools(["arxiv", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    print(agent.run(
        "what happened to the children that survieved a plane crash in colombia?"))


def main():
    agent_prediction()


if __name__ == "__main__":
    main()
