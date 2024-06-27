import json
import os

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
