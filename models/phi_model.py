from .base_model import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from threading import Thread
import guidance

from guidance import gen, select

class PhiModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            load_in_8bit=True,
            load_in_4bit=False,
            trust_remote_code=True,
            use_flash_attention_2=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       use_fast=False,
                                                       trust_remote_code=True)
        special_tokens = ['•', '¶', '∂', 'ƒ', '˙', '∆', '£', 'Ħ', '爨', 'ൠ', 'ᅘ', '∰', '፨']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        model_obj.resize_token_embeddings(len(self.tokenizer))
        self.model = guidance.models.Transformers(model_obj, self.tokenizer, echo=False)

    def perform_step(self, conversation_history):

        ## add result from previous test case here
        @guidance
        def context(lm, prev):
            newline = "\n"
            lm += f"""
            <|user|>
            {prev}
            <|end|>
            """
            return lm

        if len(conversation_history) > 1:
            self.model += context(conversation_history[-1]['content'])

        ## add next test case here
        @guidance
        def guide(lm):
            newline = '\n'
            lm += "<|assistant|>\n"
            lm += f"Reasoning: {gen('logic', stop=[newline, 'Test Case:', 'Final Guess:'])}\n"
            lm += select(["Test Case: ```(",
                          "Final Guess: ```lambda x,y,z: "])
            lm += gen("selection", stop="`")
            lm += "```\n<|end|>\n"
            return lm

        self.model += guide()
        response = str(self.model)
        start_tag = "<|assistant|>"
        end_tag = "<|end|>"
        start_index = response.rfind(start_tag)
        end_index = response.rfind(end_tag)
    
        if start_index != -1 and end_index != -1 and start_index < end_index:
            extracted_response = response[start_index + len(start_tag):end_index].strip()
            return extracted_response
        else:
            return ""

    def process_chat(self, conversation_history):
        msgs = []
        for m in conversation_history:
            msgs.append(f"<|{m['role']}|>\n{m['content']}<|end|>")
        return '\n'.join(msgs)

    def initialize_conversation(self):
        with open('./prompts/instruction.txt', 'r') as f:
            system_prompt = f.read()

        sys = [{"role": "user", "content": system_prompt}]
        chat = self.process_chat(sys)
        self.model += chat
        
        return sys
    
