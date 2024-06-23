import numpy as np
import random
import argparse
import json
import csv
import re
import openai
import os
from groq import Groq

import inspect

from tests import *

openai_api_key = os.environ["OPENAI_API_KEY"]
groq_api_key = os.environ["GROQ_API_KEY"]

GROQ_MODELS = ["llama3-70b-8192", "llama3-8b-8192", "gemma-7b-it", "mixtral-8x7b-32768"]
OPENAI_MODELS = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

class LLMReasoningHarness:
    def __init__(self, model, rule_lambda, max_attempts=30):
        self.rule = rule_lambda
        self.max_attempts = max_attempts
        self.attempts = 0
        self.test_cases = []
        self.results = []
        self.model = model

        with open('./prompts/instruction.txt', 'r') as f:
            system_prompt = f.read()

        self.conversation_history = [
            {"role": "system", "content": system_prompt}
        ]

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
        # Generate random test cases to compare the guessed lambda with the actual rule
        test_inputs = [(random.uniform(1, 100), random.uniform(1, 100), random.uniform(1, 100)) for _ in range(10000)]
        
        if all(self.rule(*inputs) == guessed_lambda(*inputs) for inputs in test_inputs):
            return "Congratulations! Your guess is correct."
        else:
            return "Sorry, that's not the correct rule."

    def model_perform_step(self):
        if self.model in GROQ_MODELS:
            client = Groq(api_key=groq_api_key)
            use_model = self.model
        else:
            openai.api_key = openai_api_key
            client = openai.OpenAI()
            use_model = self.model

        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=use_model,
                    messages=self.conversation_history
                )
            
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error {e}: Retry attempt {attempt+1}")
        return ""
    
    def interact_with_llm(self):
        while self.attempts < self.max_attempts:
            model_response = self.model_perform_step()
            print(f"LLM: {model_response}")
            
            if "test case:" in model_response.lower():
                match = re.search(r'Test Case:\s*\(([-+]?(?:\d*\.\d+|\d+))\s*,\s*([-+]?(?:\d*\.\d+|\d+))\s*,\s*([-+]?(?:\d*\.\d+|\d+))\)', model_response)

                if match:
                    numbers = tuple(map(float, match.groups()))
                    result = self.test_case(*numbers)
                else:
                    result = "Please provide exactly three numbers for the test case."

                print(f"Harness: {result}")
                self.conversation_history.append({"role": "assistant", "content": model_response})
                self.conversation_history.append({"role": "user", "content": result})

                    
            elif "final guess:" in model_response.lower():
                match = re.search(r'Final Guess:\s*(.+?)(?:\n|`|$)', model_response, re.IGNORECASE)
                if match:
                    guess_str = match.group(1).strip()
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
                        result = "Invalid lambda function. Please try again."
                        self.conversation_history.append({"role": "assistant", "content": model_response})
                        self.conversation_history.append({"role": "user", "content": result})
                        print(f"Harness: {result}")
                else:
                    result = "Unable to parse the lambda function. Please try again."
                    self.conversation_history.append({"role": "assistant", "content": model_response})
                    self.conversation_history.append({"role": "user", "content": result})
                    print(f"Harness: {result}")
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
                
def main(model):
    correct_answers = 0
    total_answers = 0
    points = 0
    attempts = []
    full_context = []
    
    for test_idx, test_rule in TESTS.items():
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
    parser.add_argument('--model', type=str, default='llama3-70b-8192',
                        help='Model to use (e.g. llama3-70b-8192)')
    args = parser.parse_args()
    main(args.model)
    
