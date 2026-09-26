# Train and save
.venv/bin/python main.py train --output checkpoints/memory-v1

# Validate: exact match, precision, recall, F1, and memory ablations
.venv/bin/python main.py validate --checkpoint checkpoints/memory-v1 --ablations

# Evaluate the test split
.venv/bin/python main.py test --checkpoint checkpoints/memory-v1
