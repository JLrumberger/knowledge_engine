import boto3

from langchain_aws import ChatBedrock # ,ChatBedrockConverse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class LlmBot:
    """
    A chatbot class that interacts with Amazon Bedrock's language models using LangChain.

    This class provides an interface for chatting with AI models, maintaining conversation
    history, and adjusting various parameters of the model.

    Attributes:
        bedrock_runtime (boto3.client): The Bedrock runtime client.
        model_id (str): The ID of the model to use.
        system_prompt (str): The system prompt that sets the context for the AI.
        messages (list): The conversation history.
        model_kwargs (dict): Additional parameters for the model.
        model (ChatBedrock): The LangChain ChatBedrock instance.

    Args:
        model_id (str, optional): The ID of the model to use. Defaults to "anthropic.claude-3-haiku-20240307-v1:0".
            Another working model that can be selected is "anthropic.claude-3-sonnet-20240229-v1:0".
        system_prompt (str, optional): The initial system prompt. Defaults to "You are a helpful AI assistant.".

    Methods:
        chat(msg): Send a message and get a response, maintaining conversation history.
        invoke(msg): Send a one-off message without affecting the conversation history.
        change_system_prompt(prompt): Change the system prompt and reset the conversation.
        get_chat_history(): Get the current chat history.
        clear_chat_history(): Clear the current chat history.
        set_temperature(temperature): Set the temperature parameter for the model.
        set_max_tokens(max_tokens): Set the maximum number of tokens for the model's response.

    Note:
        This class requires proper AWS credentials and permissions to access Amazon Bedrock services.
    """
    def __init__(self, model_id="anthropic.claude-3-haiku-20240307-v1:0", system_prompt="You are a helpful AI assistant."):
        self.bedrock_runtime = boto3.client(service_name="bedrock-runtime")
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.messages = [SystemMessage(content=self.system_prompt)]
        self.model_kwargs = {
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": ["\n\nHuman"],
        }
        self._create_model()

    def _create_model(self):
        self.model = ChatBedrock( # might need to change this to ChatBedrockConverse 
            client=self.bedrock_runtime,
            model_id=self.model_id,
            model_kwargs=self.model_kwargs,
        )

    def chat(self, msg):
        self.messages.append(HumanMessage(content=msg))
        response = self.model.invoke(self.messages)
        self.messages.append(AIMessage(content=response.content))
        return response.content

    def invoke(self, msg):
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=msg)]
        response = self.model.invoke(messages)
        return response.content

    def change_system_prompt(self, prompt):
        self.system_prompt = prompt
        self.messages = [SystemMessage(content=self.system_prompt)]
        self.model_kwargs["system"] = prompt
        self._create_model()

    def get_chat_history(self):
        return self.messages[1:]  # Exclude the system message

    def clear_chat_history(self):
        self.messages = [SystemMessage(content=self.system_prompt)]

    def set_temperature(self, temperature):
        if 0 <= temperature <= 1:
            self.model_kwargs["temperature"] = temperature
            self._create_model()
        else:
            raise ValueError("Temperature must be between 0 and 1")

    def set_max_tokens(self, max_tokens):
        if max_tokens > 0:
            self.model_kwargs["max_tokens"] = max_tokens
            self._create_model()
        else:
            raise ValueError("max_tokens must be a positive integer")

   
if __name__ == "__main__":
    import os
    os.environ['AWS_ACCESS_KEY_ID'] = ''
    os.environ['AWS_SECRET_ACCESS_KEY'] = ''
    os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'
    os.environ['AWS_SESSION_TOKEN'] = ''   
    # Create an instance
    bot = LlmBot(system_prompt="Pretend you're a helpful, talking cat. Meow!")

    # Chat with the bot
    print(bot.chat("What is 1+1?"))

    # Ask a follow-up question
    print(bot.chat("Why is that?"))

    # Use invoke for a one-off question
    print(bot.invoke("What's your favorite food?"))

    # Change the system prompt
    bot.change_system_prompt("You are a helpful math tutor, who explains the math concepts in great detail.")

    print(bot.chat("in all seriousness, what is 5+5?"))