from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms.openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

import os

os.environ["SERPER_API_KEY"] = ""
os.environ['OPENAI_API_KEY'] = ""

llm = OpenAI( temperature=0)
search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search"
    )
]

self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
prompt = lambda x : f"""Is '{x}' the name of a real town,city, or county [Yes or No] ? What county or region is the town of '{x}' in? If this is unknown, are there any towns, cities, or counties with similar names to '{x}' """

self_ask_with_search.run(prompt('Hilton'))
