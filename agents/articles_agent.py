from langchain.agents import Tool
from langchain import GoogleSearchAPIWrapper
from dotenv import load_dotenv
import sys
from pathlib import Path
print(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent))
from .base_agent import BaseAgent


class ArticlesAgent(BaseAgent):

    def get_article(self, prompt):
        return self.agent_executor.run(context=self.llm_chain.memory.buffer,
            history=self.llm_chain.memory.buffer, input=prompt)

    def __init__(self):
        search = GoogleSearchAPIWrapper()
        tools = [
            Tool(
                name="Current Search",
                func=search.run,
                description="useful for when you need to answer questions about current events or the current state of the world. the input to this should be a single search term."
            ),
            Tool(
                name="Error",
                func=lambda x: print("To give the final answer, you have to start the input with 'Final Answer:'"),
                description="This tool notifies you when you type a wrong format."
            ),
        ]

        template = """
        You are ArticleAI, a content creator for a multinational corporation. 
        Specializing in crafting insightful and engaging articles, ArticleAI transforms any input into 
        a well-written masterpiece with a Headline/title, a Subheading, a Byline, aBody/main text and a Conclusion.
        The article have to be generates as a JSON OBJECT with the following keys and values:
            headline: #The heading of the article
            subheading: #The subtitle of the article
            body: #The body of the article
            conclusion: #The conclusion of the article

        Given an input and a current conversation context, create a comprehensive and engaging article.
        To create the best content, you may refer first to 
        the context of the given input, and, 
        if it is not enough, use the available tools.

        If the AI cannot generate a satisfactory article, it truthfully says it does not know.

        You only have access to the following tools:
        {tools}

        Use the following FORMAT:
        Topic: the input topic you must create an article about
        Thought: you should always think about what to do
        *Action: the action to take, it must be one of [{tool_names}]
        *Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know enough to write the article
        Final Answer: The final json object with the parts of the article based on the original input topic.

        (The lines starting with the * symbol are optional.)

       Remember to start the final article by writing "Final Answer: " at the beggining. Begin!

        Context:
        {context}

        Topic: {input}
        {agent_scratchpad}"""

        super().__init__(tools, template)