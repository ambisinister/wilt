"""
TODO: Deprecate, doubt this will be flexible enough to be useful for all model types
"""

from .base_model import BaseModel
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import guidance

class LocalModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        # self.model = LlamaForCausalLM.from_pretrained(
        #     model_name,
        #     device_map="auto",
        #     torch_dtype=torch.float16,
        #     load_in_8bit=True,
        #     load_in_4bit=False,
        #     trust_remote_code=True,
        #     use_flash_attention_2=True
        # )
        self.model = guidance.models.Transformers(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def perform_step(self, conversation_history):
        try:
            msg_text = self.process_chat(conversation_history)
            
            #input_ids = self.tokenizer(msg_text, return_tensors="pt").input_ids.to("cuda")
            # TODO get this working to make code simpler i guess
            #input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            program = guidance('''
                {{msg_text}}
                {{#assistant~}}
                {{#select "response_type"}}
                {{#option "test"}}
                {{gen "logic" max_tokens=100}}
                Test Case:```({{gen "x" stop=","}},{{gen "y" stop=","}},{{gen "z" stop=")"}})```
                {{/option}}
                {{#option "guess"}}
                {{gen "logic" max_tokens=100}}
                Final Guess: ```lambda x,y,z:{{gen "guess" stop="```"}}```
                {{/option}}
                {{/select}}
                {{~/assistant}}
            ''')

            def generate():
                return program(msg_text=msg_text)['assistant']
            
            # generation_kwargs = {
            #     "input_ids": input_ids,
            #     "max_new_tokens": 750,
            #     "temperature": 0.8,
            #     "repetition_penalty": 1.1,
            #     "do_sample": True,
            #     "eos_token_id": self.tokenizer.eos_token_id,
            #     "streamer": streamer
            # }
            
            thread = Thread(target=generate)
            thread.start()

            response = ""
            for new_text in streamer:
                print(new_text, end='', flush=True)
                response += new_text

            thread.join()
            return response
        except Exception as e:
            print(f"Error in LocalModel: {e}")
            return ""

    def process_chat(self, conversation_history):
        msgs = []
        for m in conversation_history:
            msgs.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        msgs.append("<|im_start|>assistant")
        return '\n'.join(msgs)

    def initialize_conversation(self):
        with open('./prompts/instruction.txt', 'r') as f:
            system_prompt = f.read()

        sys = [{"role": "system", "content": system_prompt}]
        chat = self.process_chat(sys)
        self.model += chat
        
        return ""
    
