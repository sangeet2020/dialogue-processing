"""
Helper script to analyze dataset.
"""

from collections import defaultdict

from utils.util import open_file


if __name__ == "__main__":
    dataset = open_file("data/train_0.json")
    num_examples = len(dataset)
    counter = 0
    avg_text_len = 0
    max_text_len = 0
    slots_dist = defaultdict(int)

    for example in dataset:
        if "labels" not in example:
            counter += 1
        else:
            for label in example["labels"]:
                slots_dist[label["slot"]] += 1
        
        text_len = len(example["userInput"]["text"].split(' '))
        avg_text_len += text_len
        if text_len > max_text_len:
            max_text_len = text_len

    print(f"Dataset size: {num_examples}")
    print(f"No labels examples: {counter}")
    print(f"Average example text length: {round(avg_text_len/num_examples, 2)}")
    print(f"Max example text length: {max_text_len}")
    print(f"Slots distribution:\n" + "\n".join([f"{slot}: {value}" for slot, value in slots_dist.items()]))

    example_0 = dataset[0]
    text = example_0['userInput']['text']
    slot = example_0['labels'][0]['slot']
    start_index = int(example_0['labels'][0]['valueSpan']['startIndex'])
    end_index = int(example_0['labels'][0]['valueSpan']['endIndex'])
    value = text[start_index:end_index]

    print("="*50 + "\nExample 0")
    print(f"Text: {text}")
    print(f"Slot: {slot}")
    print(f"Value: {value}")
    print("="*50)
