import re
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import HumanMessage
from langchain.prompts import BaseChatPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union

class BaseAgent:
    def __init__(self, tools: List[Tool], prompt_template: str):
        self.tools = tools
        self.prompt_template = prompt_template
        self.setup_agent()

    def setup_agent(self):
        # Set up a prompt template
        class CustomPromptTemplate(BaseChatPromptTemplate):
            template: str
            tools: List[Tool]

            def format_messages(self, **kwargs) -> str:
                # Get the intermediate steps (AgentAction, Observation tuples)
                # Format them in a particular way
                intermediate_steps = kwargs.pop("intermediate_steps")
                thoughts = ""
                for action, observation in intermediate_steps:
                    thoughts += action.log
                    thoughts += f"\nObservation: {observation}\nThought: "
                # Set the agent_scratchpad variable to that value
                kwargs["agent_scratchpad"] = thoughts
                # Create a tools variable from the list of tools provided
                kwargs["tools"] = "\n".join(
                    [f"{tool.name}: {tool.description}" for tool in self.tools])
                # Create a list of tool names for the tools provided
                kwargs["tool_names"] = ", ".join(
                    [tool.name for tool in self.tools])
                formatted = self.template.format(**kwargs)
                return [HumanMessage(content=formatted)]

        prompt = CustomPromptTemplate(
            template=self.prompt_template,
            tools=self.tools,
            input_variables=[
                "input", "intermediate_steps", "history", "context"]
        )

        class CustomOutputParser(AgentOutputParser):

            def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
                # Check if agent should finish
                if "Final Answer:" in llm_output:
                    return AgentFinish(
                        # Return values is generally always a dictionary with a single `output` key
                        # It is not recommended to try anything else at the moment :)
                        return_values={"output": llm_output.split(
                            "Final Answer:")[-1].strip()},
                        log=llm_output,

                    )
                # Parse out the action and action input
                regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
                match = re.search(regex, llm_output, re.DOTALL)
                if not match:
                    return AgentAction(tool="error", tool_input="Wrong format", log=llm_output)
                action = match.group(1).strip()
                action_input = match.group(2)
                # Return the action and action input
                return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

        output_parser = CustomOutputParser()

        # LLM Options
        llm = ChatOpenAI(temperature=0)
        # llm = HuggingFaceHub(repo_id="anon8231489123/gpt4-x-alpaca-13b-native-4bit-128g",model_kwargs={"temperature": 0, "max_length": 64})
        # local llm
        # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        # local_path = 'D:/Documentos/AI/models/llama/llama.cpp/models/7B/ggml-model-q4_0.bin'
        # llm = LlamaCpp(model_path=local_path, verbose=True, n_ctx=2048)
        print("LLM created")

        self.llm_chain = LLMChain(
            llm=llm, prompt=prompt, memory=ConversationBufferMemory(input_key="input"))
        print("LLM chain created")

        tool_names = [tool.name for tool in self.tools]
        allowed_tools = list(tool_names)
        allowed_tools.remove("Error")
        agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=allowed_tools
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=self.tools, verbose=True)

    def run_chain(self, prompt):
        return self.agent_executor.run(context=self.llm_chain.memory.buffer,
            history=self.llm_chain.memory.buffer, input=prompt)

    def run_agent(self):
        while (True):
            prompt = input("User: ")
            if (prompt == "exit"):
                break
            print(self.llm_chain.memory.buffer)

            print(self.agent_executor.run(context=self.llm_chain.memory.buffer,
                  history=self.llm_chain.memory.buffer, input=prompt))
