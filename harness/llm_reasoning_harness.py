import random
import re
from .test_cases import TESTS

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

        self.conversation_history = self._initialize_conversation()

    def _initialize_conversation(self):
        if self.model.model_name not in ["claude-3-5-sonnet-20240620"]:
            role = "user" if "Phi" in self.model.model_name else "system"
            return [{"role": role, "content": self.system_prompt}]
        else:
            return [{"role": "user", "content": "Please begin."}]

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

    # placeholder
    def guess_rule(self, guessed_lambda):
        # tests
        random_inputs = [(random.uniform(1, 100), random.uniform(1, 100), random.uniform(1, 100)) for _ in range(10000)]
        grid_inputs = [(x, y, z) for x in range(-20, 21) for y in range(-20, 21) for z in range(-20, 21)]
        test_inputs = random_inputs + grid_inputs
        
        if all(self.rule(*inputs) == guessed_lambda(*inputs) for inputs in test_inputs):
            return "Congratulations! Your guess is correct."
        else:
            return "Sorry, that's not the correct rule."

    def interact_with_llm(self):
        mulligan = False
        while self.attempts <= self.max_attempts:
            model_response = self.model.perform_step(self.conversation_history)
            print(f"LLM: {model_response}")
            
            if "test case:" in model_response.lower():
                result = self._handle_test_case(model_response, mulligan)
                if result == "ABORT":
                    return {'points': 0, 'guesses': self.attempts}
                mulligan = result == "MULLIGAN"
            elif "final guess:" in model_response.lower():
                return self._handle_final_guess(model_response)
            else:
                print("Harness: Doesn't seem like a guess or a test, retrying.")

        return {'points': 0, 'guesses': self.attempts}

    def _handle_test_case(self, model_response, mulligan):
        match = re.search(r'Test Case:\s*`+\(([-+]?(?:\d*\.\d+|\d+))\s*,\s*([-+]?(?:\d*\.\d+|\d+))\s*,\s*([-+]?(?:\d*\.\d+|\d+))\)`+', model_response)

        if match:
            numbers = tuple(map(float, match.groups()))
            result = self.test_case(*numbers)
            self._update_conversation(model_response, result)
            return False
        elif "```" not in model_response and not mulligan:
            result = "Please make sure your test and final guess are wrapped in backticks as instructed, e.g. Test Case: ```(0,0,0)``` and Final Guess: ```lambda x: x```"
            self._update_conversation(model_response, result)
            return "MULLIGAN"
        elif mulligan:
            print("Harness: Still not following instructions, aborting")
            return "ABORT"
        else:
            result = "Please provide exactly three numbers for the test case."
            self._update_conversation(model_response, result)
            return False

    def _handle_final_guess(self, model_response):
        match = re.search(r'Final Guess:\s*`+(.+?)`+', model_response, re.DOTALL)
        if match:
            guess_str = match.group(1).strip()
            try:
                guessed_lambda = eval(guess_str)
                result = self.guess_rule(guessed_lambda)
                self._update_conversation(model_response, result)
            
                if 'congratulations' in result.lower():
                    return {'points': 1000 + self.bonus_points(), 'guesses': self.attempts}
                else:
                    return {'points': 0, 'guesses': self.attempts}
            except:
                result = "Invalid lambda function."
                self._update_conversation(model_response, result)
                return {'points': 0, 'guesses': self.attempts}
        else:
            result = "Unable to parse the lambda function."
            self._update_conversation(model_response, result)
            return {'points': 0, 'guesses': self.attempts}

    def _update_conversation(self, model_response, result):
        self.conversation_history.append({"role": "assistant", "content": model_response})
        self.conversation_history.append({"role": "user", "content": result})
        print(f"Harness: {result}")
