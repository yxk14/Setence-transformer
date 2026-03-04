# Setence-transformer

This repository provides a simple Gradio interface for encoding sentences using a BERT-based
sentence embedding model (via the `sentence-transformers` library).

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
python app.py
```

3. A local Gradio web UI will open. Enter a sentence and you will see the
embedding vector and its shape/length.
