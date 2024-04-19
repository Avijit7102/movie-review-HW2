import os
import json
import sys
import string
def remove_punctuation(text):
    return ''.join([char for char in text if char not in string.punctuation]).split()


def count_frequencies(words):
    def count_frequencies(text):
        freq = {}
        for word in text:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1
        return freq


def preprocess(directory):
    print("Inside preprocess function:", directory)
    if not os.path.exists(directory):
        print("Directory does not exist:", directory)
        return
    for label in os.listdir(directory):
        feature_vectors = []
        vocabulary = set()
        for label in os.listdir(directory):
            folder = os.path.join(directory, label)
            if os.path.isdir(folder):
                for filename in os.listdir(folder):
                    if filename.endswith(".txt"):
                        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                            words = remove_punctuation(f.read())
                            words = remove_punctuation(f.read())
                            for word in words:
                                vocabulary.add(word)
                            feature_vectors.append((label, count_frequencies(words)))

    output_name = "movie-review-" + os.path.basename(directory) + "-BOW.NB"
    with open(output_name, "w") as output:
        for label, vector in feature_vectors:
            output.write(f"{label}\t{json.dumps(vector)}\n")

directory = sys.argv[1]
preprocess(directory)
