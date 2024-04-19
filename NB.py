import os
import json
import sys
import string

def remove_punctuation(text):
    return ''.join([char for char in text if char not in string.punctuation]).split()

def count_frequencies(words):
    freq = {}
    for word in words:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    return freq

def train(train_file, model_file):
    print("Training on corpus in:", train_file)
    vocabulary = set()
    class_counts = {}
    feature_counts = {}
    with open(train_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            label = parts[0]
            if label not in class_counts:
                class_counts[label] = 0
                feature_counts[label] = {}
            words = remove_punctuation(" ".join(parts[1:]))
            for word in words:
                vocabulary.add(word)
                if word in feature_counts[label]:
                    feature_counts[label][word] += 1
                else:
                    feature_counts[label][word] = 1
            class_counts[label] += 1

    # Calculate probabilities
    label_probabilities = {label: class_counts[label] / sum(class_counts.values()) for label in class_counts}
    feature_probabilities = {label: {word: (feature_counts[label].get(word, 0) + 1) / (class_counts[label] + len(vocabulary)) for word in vocabulary} for label in class_counts}

    # Save model to file
    with open(model_file, "w") as model_file:
        model_data = {
            "label_probabilities": label_probabilities,
            "feature_probabilities": feature_probabilities
        }
        model_file.write(json.dumps(model_data))

def test(test_file, model_file, output_file):
    print("Testing on corpus in:", test_file)
    with open(model_file, "r") as model_file:
        model_data = json.load(model_file)
        label_probabilities = model_data["label_probabilities"]
        feature_probabilities = model_data["feature_probabilities"]

    correct = 0
    total = 0
    with open(test_file, "r") as test_file, open(output_file, "w") as output_file:
        for line in test_file:
            parts = line.strip().split()
            true_label = parts[0]
            document = " ".join(parts[1:])
            # Calculate probabilities for each class
            document_words = remove_punctuation(document)
            class_probabilities = {label: label_probabilities[label] for label in label_probabilities}
            for label in label_probabilities:
                for word in document_words:
                    if word in feature_probabilities[label]:
                        class_probabilities[label] *= feature_probabilities[label][word]
            # Determine most likely class
            predicted_label = max(class_probabilities, key=class_probabilities.get)
            output_file.write(f"{predicted_label}\t{json.dumps(class_probabilities)}\n")
            if predicted_label == true_label:
                correct += 1
            total += 1

    accuracy = correct / total
    output_file.write(f"Accuracy: {accuracy}\n")
    print("Accuracy:", accuracy)

# Main
if len(sys.argv) != 5:
    print("Usage: python NB.py <train_file> <test_file> <model_file> <output_file>")
    sys.exit(1)

train_file = sys.argv[1]
test_file = sys.argv[2]
model_file = sys.argv[3]
output_file = sys.argv[4]

train(train_file, model_file)
test(test_file, model_file, output_file)
