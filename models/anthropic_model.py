from .base_model import BaseModel
from anthropic import Anthropic
import os

class AnthropicModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
        self.client = Anthropic(api_key=anthropic_api_key)

    def perform_step(self, conversation_history):
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=conversation_history
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error in AnthropicModel: {e}")
            return ""
