"""
Deprecated, will delete once everything is confirmed working again
"""

#standard lib
import numpy as np
import random
import argparse
import json
import csv
import re
import os
import inspect

#apis
import openai
from groq import Groq
from anthropic import Anthropic

#local
import torch
from transformers import LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer
from threading import Thread
import flash_attn, bitsandbytes

#in proj
from tests import *

torch.random.manual_seed(0)

openai_api_key = os.environ["OPENAI_API_KEY"]
groq_api_key = os.environ["GROQ_API_KEY"]
deepseek_api_key = os.environ["DEEPSEEK_API_KEY"]
anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]

GROQ_MODELS = ["llama3-70b-8192", "llama3-8b-8192", "gemma-7b-it", "mixtral-8x7b-32768"]
OPENAI_MODELS = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-coder"]
ANTHROPIC_MODELS = ["claude-3-5-sonnet-20240620"]

LOCAL_MODELS = ["microsoft/Phi-3-mini-4k-instruct", "NousResearch/Hermes-2-Theta-Llama-3-8B"]

class LLMReasoningHarness:
    def __init__(self, model, rule_lambda, max_attempts=30):
        self.rule = rule_lambda
        self.max_attempts = max_attempts
        self.attempts = 0
        self.test_cases = []
        self.results = []
        self.model = model

        with open('./prompts/instruction.txt', 'r') as f:
            self.system_prompt = f.read()

        if self.model not in ANTHROPIC_MODELS:
            if "Phi" in self.model:
                this_role = "user"
            else:
                this_role = "system"
            self.conversation_history = [
                {"role": this_role, "content": self.system_prompt}
            ]
        else:
            self.conversation_history = [{"role": "user", "content": "Please begin."}]

    def test_case(self, x, y, z):
        if self.attempts >= self.max_attempts:
            return "Maximum attempts reached. Please make a guess."
        
        result = self.rule(x, y, z)
        self.attempts += 1
        self.test_cases.append((x, y, z))
        self.results.append(result)
        att_remaining = self.max_attempts - self.attempts
        
        return f"Result for input ({x}, {y}, {z}): {result}. Attempts Remaining: {att_remaining}"

    def bonus_points(self):
        return 100 * (1 - (self.attempts / self.max_attempts))

    ## placeholder
    def guess_rule(self, guessed_lambda):
        # tests
        random_inputs = [(random.uniform(1, 100), random.uniform(1, 100), random.uniform(1, 100)) for _ in range(10000)]
        grid_inputs = [(x, y, z) for x in range(-20, 21) for y in range(-20, 21) for z in range(-20, 21)]
        test_inputs = random_inputs + grid_inputs
        
        if all(self.rule(*inputs) == guessed_lambda(*inputs) for inputs in test_inputs):
            return "Congratulations! Your guess is correct."
        else:
            return "Sorry, that's not the correct rule."

    def process_chat(self):
        msgs = []
        for m in self.conversation_history:
            msgs.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
            msgs.append("<|im_start|>assistant")
        return '\n'.join(msgs)

    def model_perform_step(self):
        if self.model in GROQ_MODELS:
            client = Groq(api_key=groq_api_key)
            use_model = self.model
        elif self.model in ANTHROPIC_MODELS:
            client = Anthropic(api_key=anthropic_api_key)
            use_model = self.model
        elif self.model in DEEPSEEK_MODELS:
            client = openai.OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
            use_model = self.model
        elif self.model in LOCAL_MODELS:
            model = LlamaForCausalLM.from_pretrained(
                self.model,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=True,
                load_in_4bit=False,
                trust_remote_code=True,
                use_flash_attention_2=True
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model)
        else:
            openai.api_key = openai_api_key
            client = openai.OpenAI()
            use_model = self.model

        max_retries = 3

        for attempt in range(max_retries):
            try:
                if self.model in LOCAL_MODELS:
                    msg_text = self.process_chat()
                    input_ids = tokenizer(msg_text, return_tensors="pt").input_ids.to("cuda")
                    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    generation_kwargs = {
                        "input_ids": input_ids,
                        "max_new_tokens": 750,
                        "temperature": 0.8,
                        "repetition_penalty": 1.1,
                        "do_sample": True,
                        "eos_token_id": tokenizer.eos_token_id,
                        "streamer": streamer
                    }
                    
                    thread = Thread(target=model.generate, kwargs=generation_kwargs)
                    thread.start()

                    response = ""
                    for new_text in streamer:
                        print(new_text, end='', flush=True)
                        response += new_text

                    thread.join()
                    return response[len(msg_text):].strip()
                    
                if self.model in ANTHROPIC_MODELS:
                    messages = [{"role": msg["role"], "content": msg["content"]} for msg in self.conversation_history]
                    response = client.messages.create(
                        model=use_model,
                        max_tokens=1024,
                        system=self.system_prompt,
                        messages=messages
                    )
                    return response.content[0].text
                else:
                    response = client.chat.completions.create(
                        model=use_model,
                        messages=self.conversation_history
                    )
                    return response.choices[0].message.content
            except Exception as e:
                print(f"Error {e}: Retry attempt {attempt+1}")
        return ""
    
    def interact_with_llm(self):
        mulligan = False
        while self.attempts <= self.max_attempts:
            model_response = self.model_perform_step()
            print(f"LLM: {model_response}")
            
            if "test case:" in model_response.lower():
                match = re.search(r'Test Case:\s*`+\(([-+]?(?:\d*\.\d+|\d+))\s*,\s*([-+]?(?:\d*\.\d+|\d+))\s*,\s*([-+]?(?:\d*\.\d+|\d+))\)`+', model_response)

                if match:
                    numbers = tuple(map(float, match.groups()))
                    result = self.test_case(*numbers)
                    mulligan = False
                elif "```" not in model_response and not mulligan:
                    result = "Please make sure your test and final guess are wrapped in backticks as instructed, e.g. Test Case: ```(0,0,0)``` and Final Guess: ```lambda x: x```"
                    mulligan = True
                elif mulligan:
                    print("Harness: Still not following instructions, aborting")
                    return {'points': 0, 'guesses': self.attempts}
                else:
                    result = "Please provide exactly three numbers for the test case."

                print(f"Harness: {result}")
                self.conversation_history.append({"role": "assistant", "content": model_response})
                self.conversation_history.append({"role": "user", "content": result})

                    
            elif "final guess:" in model_response.lower():
                match = re.search(r'Final Guess:\s*`+(.+?)`+', model_response, re.DOTALL)
                if match:
                    guess_str = match.group(1).strip()
                    print(guess_str)
                    try:
                        guessed_lambda = eval(guess_str)
                        result = self.guess_rule(guessed_lambda)
                        self.conversation_history.append({"role": "assistant", "content": model_response})
                        self.conversation_history.append({"role": "user", "content": result})
                        print(f"Harness: {result}")
                        
                        if 'congratulations' in result.lower():
                            return {'points': 1000 + self.bonus_points(), 'guesses': self.attempts}
                        else:
                            return {'points': 0, 'guesses': self.attempts}
                    except:
                        result = "Invalid lambda function."
                        self.conversation_history.append({"role": "assistant", "content": model_response})
                        self.conversation_history.append({"role": "user", "content": result})
                        print(f"Harness: {result}")
                        return {'points': 0, 'guesses': self.attempts}
                else:
                    result = "Unable to parse the lambda function."
                    self.conversation_history.append({"role": "assistant", "content": model_response})
                    self.conversation_history.append({"role": "user", "content": result})
                    print(f"Harness: {result}")
                    return {'points': 0, 'guesses': self.attempts}
            else:
                print("Harness: Doesn't seem like a guess or a test, retrying.")

def dump_results(model_name, accuracy, avg_guesses, points, full_context):
    os.makedirs('./results', exist_ok=True)
    
    csv_file = f'./results/{model_name}_results.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['accuracy', 'avg_guesses', 'points'])
        writer.writerow([accuracy, avg_guesses, points])
    
    json_file = f'./results/{model_name}_full_context.json'
    with open(json_file, 'w') as f:
        json.dump(full_context, f, indent=2)

def save_checkpoint(model_name, current_test_idx, correct_answers, total_answers, points, attempts, full_context):
    checkpoint = {
        'current_test_idx': current_test_idx,
        'correct_answers': correct_answers,
        'total_answers': total_answers,
        'points': points,
        'attempts': attempts,
        'full_context': full_context
    }
    
    checkpoint_file = f'./checkpoints/{model_name}_checkpoint.json'
    os.makedirs('./checkpoints', exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def load_checkpoint(model_name):
    checkpoint_file = f'./checkpoints/{model_name}_checkpoint.json'
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

                
def main(model):
    checkpoint = load_checkpoint(model)
    if checkpoint:
        print("Resuming from checkpoint...")
        current_test_idx = checkpoint['current_test_idx']
        correct_answers = checkpoint['correct_answers']
        total_answers = checkpoint['total_answers']
        points = checkpoint['points']
        attempts = checkpoint['attempts']
        full_context = checkpoint['full_context']
    else:
        current_test_idx = 1
        correct_answers = 0
        total_answers = 0
        points = 0
        attempts = []
        full_context = []

    num_tests = len(TESTS)

    for test_idx in range(current_test_idx, num_tests + 1):
        test_rule = TESTS[str(test_idx)]
        print(inspect.getsource(test_rule))
        harness = LLMReasoningHarness(model=model, rule_lambda=test_rule)
        test_success = harness.interact_with_llm()

        if test_success['points'] != 0:
            correct_answers += 1
        points += test_success['points']
        total_answers += 1
        attempts.append(test_success['guesses'])

        full_context.append({
            'test_index': test_idx,
            'conversation_history': harness.conversation_history,
            'points': test_success['points'],
            'guesses': test_success['guesses']
        })

        save_checkpoint(model, test_idx+1, correct_answers,
                        total_answers, points, attempts, full_context)

    acc = correct_answers/total_answers
    avg_guesses = np.mean(attempts)

    print("===========")
    print("Final Result")
    print(f"Accuracy: {correct_answers} / {total_answers} = {acc}")
    print(f"Average Tests = {avg_guesses}")
    print(f"Total Points = {points}")

    dump_results(model, acc, avg_guesses, points, full_context)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLM Reasoning Harness")
    parser.add_argument('--model', type=str, default="NousResearch/Hermes-2-Theta-Llama-3-8B",
                        help='Model to use (e.g. llama3-70b-8192)')
    args = parser.parse_args()
    main(args.model)
    
