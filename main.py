import argparse
import numpy as np
from harness.llm_reasoning_harness import LLMReasoningHarness
from models.model_factory import ModelFactory
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.result_handler import dump_results
from harness.test_cases import TESTS

def main(model_name):
    model = ModelFactory.create_model(model_name)
    checkpoint = load_checkpoint(model_name)
    
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
        print(f"Running test {test_idx}")
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

        save_checkpoint(model_name, test_idx+1, correct_answers,
                        total_answers, points, attempts, full_context)

    acc = correct_answers/total_answers
    avg_guesses = np.mean(attempts)

    print("===========")
    print("Final Result")
    print(f"Accuracy: {correct_answers} / {total_answers} = {acc}")
    print(f"Average Tests = {avg_guesses}")
    print(f"Total Points = {points}")

    dump_results(model_name, acc, avg_guesses, points, full_context)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLM Reasoning Harness")
    parser.add_argument('--model', type=str, default="microsoft/Phi-3-mini-4k-instruct",
                        help='Model to use (e.g. llama3-70b-8192)')
    args = parser.parse_args()
    main(args.model)
