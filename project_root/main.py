from pathlib import Path

def load_raw_text(data_dir="data"):
    data_path = Path(data_dir)
    texts = []
    
    for file_path in data_path.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    
    
    return "\n\n".join(texts)


if __name__ == "__main__":
    raw_text = load_raw_text()
    print("Total characters:", len(raw_text))
    print(raw_text[:500])