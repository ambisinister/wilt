#from .openai_model import OpenAIModel
from .groq_model import GroqModel
#from .anthropic_model import AnthropicModel
#from .deepseek_model import DeepseekModel
from .hermes_model import LocalModel

class ModelFactory:
    @staticmethod
    def create_model(model_name):
        if model_name in ["llama3-70b-8192", "llama3-8b-8192", "gemma-7b-it", "mixtral-8x7b-32768"]:
            return GroqModel(model_name)
        elif model_name in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
            return OpenAIModel(model_name)
        elif model_name in ["claude-3-5-sonnet-20240620"]:
            return AnthropicModel(model_name)
        elif model_name in ["deepseek-chat", "deepseek-coder"]:
            return DeepseekModel(model_name)
        elif model_name in ["microsoft/Phi-3-mini-4k-instruct", "NousResearch/Hermes-2-Theta-Llama-3-8B"]:
            return LocalModel(model_name)
        else:
            raise ValueError(f"Unknown model: {model_name}")
