import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
from pathlib import Path
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)
np.random.seed(42)
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
def self_attention(X):
    d = X.shape[1]
    Wq = np.random.randn(d, d) * 0.2
    Wk = np.random.randn(d, d) * 0.2
    Wv = np.random.randn(d, d) * 0.2
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    scores = Q @ K.T / np.sqrt(d)
    attention = softmax(scores)
    output = attention @ V
    return attention, output
def multi_head_attention(X, num_heads=2):
    sequence_length, dimension = X.shape
    head_dimension = dimension // num_heads
    if dimension % num_heads != 0:
        raise ValueError("Dimension must divide evenly into heads.")
    outputs = []
    attentions = []
    for head in range(num_heads):
        Wq = np.random.randn(dimension, head_dimension) * 0.2
        Wk = np.random.randn(dimension, head_dimension) * 0.2
        Wv = np.random.randn(dimension, head_dimension) * 0.2
        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv
        scores = Q @ K.T / np.sqrt(head_dimension)
        attention = softmax(scores)
        head_output = attention @ V
        outputs.append(head_output)
        attentions.append(attention)
    concatenated = np.concatenate(outputs, axis=1)
    Wo = np.random.randn(
        num_heads * head_dimension,
        dimension
    ) * 0.2
    projected = concatenated @ Wo
    return attentions, concatenated, projected
def positional_encoding(position, dimension):
    encoding = np.zeros((position, dimension))
    for pos in range(position):
        for i in range(dimension):
            angle = pos / (10000 ** ((2 * (i // 2)) / dimension))
            if i % 2 == 0:
                encoding[pos, i] = np.sin(angle)
            else:
                encoding[pos, i] = np.cos(angle)
    return encoding
def task3():
    print("\nTRANSFORMER CONCEPTS")
    X = np.random.randn(4, 8)
    print("\n1. SINGLE-HEAD SELF ATTENTION")
    attention, output = self_attention(X)
    print("\nAttention weights (4x4):")
    print(np.round(attention, 4))
    print("\nRow sums:")
    print(np.sum(attention, axis=1))
    assert np.allclose(
        np.sum(attention, axis=1),
        np.ones(4)
    )
    print("Every row sums to 1.0")
    plt.figure(figsize=(6, 5))
    plt.imshow(attention, cmap="viridis")
    plt.colorbar(label="Attention Weight")
    plt.title("Single-Head Attention Matrix")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.savefig(OUTPUT / "attention_matrix.png",dpi=150)
    plt.close()
    print("\n2. MULTI-HEAD ATTENTION")
    heads, concatenated, projected = multi_head_attention(X)
    for i, head in enumerate(heads, 1):
        print(f"\nHead {i}:")
        print(np.round(head, 4))
    print("\nConcatenated Shape:", concatenated.shape)
    print("\nOutput Projection:")
    print(np.round(projected, 4))
    print("\n3. POSITIONAL ENCODING")
    pe = positional_encoding(10, 8)
    print(np.round(pe, 4))
    plt.figure(figsize=(8, 5))
    plt.imshow(pe, cmap="viridis", aspect="auto")
    plt.colorbar(label="Encoding Value")
    plt.title("Sinusoidal Positional Encoding")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.savefig(OUTPUT / "positional_encoding.png",dpi=150)
    plt.close()
    print("Positional encoding plot saved.")
    print("\n4. HUGGING FACE BERT FILL-MASK")
    try:
        fill_mask = pipeline(
            "fill-mask",
            model="google-bert/bert-base-uncased"
        )
        result = fill_mask(
            "Artificial Intelligence is [MASK]."
        )
        print("Input: Artificial Intelligence is [MASK].")
        for item in result[:3]:
            print(item["sequence"], item["score"])
    except Exception as error:
        print("BERT error:", error)
    print("\n5. HUGGING FACE GPT-2")
    try:
        generator = pipeline(
            "text-generation",
            model="distilgpt2"
        )
        result = generator(
            "Artificial Intelligence is",
            max_new_tokens=40,
            num_return_sequences=1,
            pad_token_id=50256
        )
        print(result[0]["generated_text"])
    except Exception as error:
        print("GPT-2 error:", error)
    print("\n6. TRANSFORMER THEORY")
    explanations = [
        "Encoder-only models such as BERT understand context and are useful for classification.",
        "Decoder-only models such as GPT generate text autoregressively.",
        "Encoders use bidirectional context to represent input tokens.",
        "Decoders use causal attention to predict the next token.",
        "Transformers process token relationships through attention rather than recurrent hidden states.",
        "They support parallel training and capture long-range dependencies more effectively than traditional RNNs."
    ]
    for line in explanations:
        print("-", line)

if __name__ == "__main__":
    task3()
