"""
TODO: Deprecate, doubt this will be flexible enough to be useful for all model types
"""

from .base_model import BaseModel
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import guidance

from guidance import gen, select

class LocalModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        model_obj = LlamaForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            load_in_4bit=False,
            trust_remote_code=True,
            use_flash_attention_2=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)        
        self.model = guidance.models.Transformers(model_obj, self.tokenizer, echo=False)

    def perform_step(self, conversation_history):

        ## add result from previous test case here
        @guidance
        def context(lm, prev):
            newline = "\n"
            lm += f"""
            <|im_start|>user
            {prev}
            <|im_end|>
            """
            return lm

        if conversation_history[-1]['role'] == "user":
            self.model += context(conversation_history[-1]['content'])

        ## add next test case here
        ## TODO: Don't let it yap after the test case
        @guidance
        def guide(lm):
            lm += gen('logic', stop='\n')
            lm += select([
                f"Test Case:```({gen('x', stop=',')},{gen('y', stop=',')},{gen('z', stop=')')})`",
                f"Final Guess: ```lambda x,y,z:{gen('guess', stop='`')}`"
                ])
            return lm

        self.model += guide()
        response = str(self.model)

        return response

            
        try:
            return str(self.model)
        except Exception as e:
            print(f"Error in LocalModel: {e}")
            return ""

    def process_chat(self, conversation_history):
        msgs = []
        for m in conversation_history:
            msgs.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        return '\n'.join(msgs)

    def initialize_conversation(self):
        with open('./prompts/instruction.txt', 'r') as f:
            system_prompt = f.read()

        sys = [{"role": "system", "content": system_prompt}]
        chat = self.process_chat(sys)
        self.model += chat
        
        return sys
    
