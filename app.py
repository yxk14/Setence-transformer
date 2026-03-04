from sentence_transformers import SentenceTransformer
import gradio as gr
import numpy as np

# Load a pre-trained BERT-based sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


def encode_sentence(sentence: str):
    """Encode the input sentence and return the embedding vector and its shape.

    Args:
        sentence (str): A single input sentence.

    Returns:
        tuple: (str representation of embedding, str of shape)
    """
    if sentence is None or sentence.strip() == "":
        return "", "(0,)"

    # Compute embeddings
    emb = model.encode(sentence)
    # Convert embedding to list for pretty printing
    emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
    shape = emb.shape if hasattr(emb, "shape") else (len(emb),)
    return str(emb_list), str(shape)


iface = gr.Interface(
    fn=encode_sentence,
    inputs=gr.Textbox(lines=2, placeholder="Enter a sentence here...", label="Input Sentence"),
    outputs=[
        gr.Textbox(label="Embedding Vector"),
        gr.Textbox(label="Embedding Shape/Length"),
    ],
    title="Sentence Transformer Encoder",
    description="Type a sentence and get its BERT-based sentence embedding vector along with its shape.",
)

if __name__ == "__main__":
    iface.launch()