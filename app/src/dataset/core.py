import random

def generate_binary_sequence(dataset_size, row_size):
    dataset = []
    for _ in range(dataset_size):
        row = []
        for i in range(row_size):
            row.append(random.randint(0, 1))
        
        dataset.append(row)
    
    return dataset