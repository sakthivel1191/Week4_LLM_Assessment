import tiktoken
import pandas as pd
from pathlib import Path
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)
encoding = tiktoken.get_encoding("cl100k_base")
models = [
    {
        "Family": "GPT",
        "Owner": "OpenAI",
        "Weights": "Closed",
        "Context": "Up to 1M tokens (GPT-4.1)",
        "Strengths": "Reasoning, coding, instruction following",
        "Source": "https://developers.openai.com/api/docs/models/gpt-4.1"
    },
    {
        "Family": "Claude",
        "Owner": "Anthropic",
        "Weights": "Closed",
        "Context": "200K or 1M depending on model",
        "Strengths": "Writing, reasoning, long documents",
        "Source": "https://platform.claude.com/docs/en/build-with-claude/context-windows"
    },
    {
        "Family": "Llama",
        "Owner": "Meta",
        "Weights": "Open-weight",
        "Context": "4K to 10M depending on release",
        "Strengths": "Self-hosting, customization, privacy",
        "Source": "https://github.com/meta-llama/llama-models"
    },
    {
        "Family": "Gemini",
        "Owner": "Google",
        "Weights": "Closed",
        "Context": "Up to 1M tokens (Gemini 2.5 Flash)",
        "Strengths": "Multimodal tasks, long context, speed",
        "Source": "https://ai.google.dev/gemini-api/docs/models"
    }
]
texts = {
    "Sentence": (
        "Large language models can understand human language."
    ),
    "Paragraph": (
        "Artificial Intelligence is transforming industries. "
        "It helps businesses automate repetitive tasks, "
        "improve productivity, and analyze large amounts of data."
    ),
    "Page": (
        "Artificial Intelligence is an important field of computer science. "
        "Machine learning allows systems to learn from data. "
        "Deep learning uses neural networks with multiple layers. "
        "Natural language processing enables computers to understand text. "
        "Transformers are widely used in modern language models. "
        "These systems can generate text, summarize documents, "
        "answer questions, translate languages, and classify information. "
    ) * 12,
    "JSON": (
        '{"name":"Anita","role":"Data Scientist",'
        '"skills":["Python","SQL","Machine Learning"],'
        '"experience":3,"location":"Chennai"}'
    ),
    "Tamil": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?"
}
def estimate_cost(
    input_tokens,
    output_tokens,
    price_in,
    price_out
):
    """
    Prices are USD per 1 million tokens.
    """
    input_cost = input_tokens * price_in / 1_000_000
    output_cost = output_tokens * price_out / 1_000_000
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost
    }
def task1():
    print("\nMODELS, TOKENS AND COST")
    print("\n1. MODEL FAMILY COMPARISON")
    for model in models:
        print("\n", model)
    pd.DataFrame(models).to_csv(OUTPUT / "model_comparison.csv",index=False)
    print("\n2. TOKEN COUNTING")
    results = []
    for name, text in texts.items():
        tokens = encoding.encode(text)
        token_count = len(tokens)
        words = len(text.split())
        words_per_token = (words / token_count if token_count else 0)
        result = {
            "Text Type": name,
            "Words": words,
            "Tokens": token_count,
            "Words Per Token": round(words_per_token, 4),
            "Fits 8K": token_count <= 8192,
            "Fits 128K": token_count <= 131072
        }
        results.append(result)
        print("\n", name)
        print("Words:", words)
        print("Tokens:", token_count)
        print("Words per token:", round(words_per_token, 4))
        print("Fits 8K:", token_count <= 8192)
        print("Fits 128K:", token_count <= 131072)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT / "token_results.csv",index=False)
    print("\n3. COST ESTIMATION")
    price_in = 2.00
    price_out = 8.00
    daily_chats = 1000
    monthly_days = 30
    input_per_chat = 1500
    output_per_chat = 400
    daily_input = daily_chats * input_per_chat
    daily_output = daily_chats * output_per_chat
    monthly_input = daily_input * monthly_days
    monthly_output = daily_output * monthly_days
    monthly_cost = estimate_cost(
        monthly_input,
        monthly_output,
        price_in,
        price_out
    )
    print("\nInput tokens per month:", monthly_input)
    print("Output tokens per month:", monthly_output)
    print("Monthly input cost:",f"${monthly_cost['input_cost']:.4f}")
    print("Monthly output cost:",f"${monthly_cost['output_cost']:.4f}")
    print("Monthly total cost:",f"${monthly_cost['total_cost']:.4f}")
    print("\n4. MODEL RECOMMENDATIONS")
    print("""
    Public customer-support bot:
    Recommend GPT-4.1 or a suitable current GPT model.
    It offers strong instruction following and tool support.
    A managed API avoids maintaining GPU infrastructure.
    Apply privacy controls and human escalation.
    Evaluate cost and accuracy before production.
    Private on-premise tool:
    Recommend Meta Llama open-weight models.
    Model weights can be deployed on private infrastructure.
    Sensitive data can remain within the organization.
    The organization controls serving and customization.
    Hardware, license and maintenance costs must be considered.
    """)

if __name__ == "__main__":
    task1()
    