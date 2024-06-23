import random
import re
import openai
import os
from groq import Groq

import inspect

from tests import *

openai_api_key = os.environ["OPENAI_API_KEY"]
groq_api_key = os.environ["GROQ_API_KEY"]

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

    ## placeholder
    def guess_rule(self, guessed_lambda):
        # Generate random test cases to compare the guessed lambda with the actual rule
        test_inputs = [(random.uniform(1, 100), random.uniform(1, 100), random.uniform(1, 100)) for _ in range(10000)]
        
        if all(self.rule(*inputs) == guessed_lambda(*inputs) for inputs in test_inputs):
            return "Congratulations! Your guess is correct."
        else:
            return "Sorry, that's not the correct rule."

    def model_perform_step(self):
        if self.model == 'llama3-70b-8192':
            client = Groq(api_key=groq_api_key)
            use_model = "llama3-70b-8192"
        else:
            openai.api_key = openai_api_key
            client = openai.OpenAI()
            use_model = "gpt-4o"
            
        response = client.chat.completions.create(
            model=use_model,
            messages=self.conversation_history
        )

        return response.choices[0].message.content
    
    def interact_with_llm(self):
        while self.attempts < self.max_attempts:
            model_response = self.model_perform_step()
            print(f"LLM: {model_response}")
            
            if "test case:" in model_response.lower():
                match = re.search(r'Test Case:\s*\(([-+]?(?:\d*\.\d+|\d+))\s*,\s*([-+]?(?:\d*\.\d+|\d+))\s*,\s*([-+]?(?:\d*\.\d+|\d+))\)', model_response)

                if match:
                    numbers = tuple(map(float, match.groups()))
                    result = self.test_case(*numbers)
                    self.conversation_history.append({"role": "assistant", "content": model_response})
                    self.conversation_history.append({"role": "user", "content": result})
                    print(f"Harness: {result}")
                else:
                    print("Harness: Please provide exactly three numbers for the test case.")
                    
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
                            return True
                        else:
                            return False
                    except:
                        print("Harness: Invalid lambda function. Please try again.")
                else:
                    print("Harness: Unable to parse the lambda function. Please try again.")
            else:
                print("Harness: Doesn't seem like a guess or a test.")
    
if __name__ == '__main__':
    correct_answers = 0
    total_answers = 0
    
    for test_idx, test_rule in TESTS.items():
        print(inspect.getsource(test_rule))
        harness = LLMReasoningHarness(model='llama3-70b-8192', rule_lambda=test_rule)
        test_success = harness.interact_with_llm()

        if test_success:
            correct_answers += 1
        total_answers += 1

    print(f"Final Result: {correct_answers} / {total_answers}")
