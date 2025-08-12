import os
from sklearn.model_selection import train_test_split

def check_file(path: str) -> list[str]:
    relative_path = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".txt") and os.path.isfile(os.path.join(path, f))]
    return relative_path

def concatenate_content(path: str) -> str:
    concatenate_file = "new_corpus.txt"
    total_file = check_file(path)
    if len(total_file) > 1:
        for file in total_file:
            try:
                with open(file, encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file, encoding="latin1") as f:
                    content = f.read()
            with open(concatenate_file, "a") as f:
                f.write(content)
        for file in total_file:
            os.remove(file)
        return os.path.join(path, concatenate_file)
    return total_file[0]

def read_data(directory: str) -> str:
    full_text = ""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, encoding="utf-8") as f:
                full_text += f.read() + "\n"
    return full_text
        
def load_dataset(sequences):
    X = sequences[:, :-1]
    y = sequences[:, 1:]
    return X, y