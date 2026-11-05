import pandas as pd
from pathlib import Path
from gemini_helper import ask_gemini
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)
results = []
def run_test(name, prompt, rating=None):
    print(f"\n{name}")
    print("Prompt:", prompt)
    answer, error = ask_gemini(prompt)
    if error:
        answer = "NOT RUN: " + error
        actual_rating = None
    else:
        print("Result:", answer)
        actual_rating = rating
    results.append({
        "Test": name,
        "Prompt": prompt,
        "Result": answer,
        "Rating": actual_rating
    })
    return answer
def task2():
    print("\nCAPABILITIES AND LIMITATIONS")
    run_test(
        "Summarization",
        "Summarize in two sentences: "
        "AI is used in healthcare, education, banking and transport."
    )
    run_test(
        "Classification",
        "Classify sentiment as Positive, Negative or Neutral: "
        "I love this excellent product."
    )
    run_test(
        "Extraction",
        "Extract name, skills and experience from: "
        "Ravi is a Python developer with 4 years of SQL experience. "
        "Return JSON."
    )
    run_test(
        "Translation",
        "Translate into Tamil: Artificial Intelligence is useful."
    )
    run_test(
        "Code Generation",
        "Write a Python function to find the largest of three numbers."
    )
    run_test(
        "Knowledge Cutoff",
        "What is the latest major technology announcement "
        "on today's date? State uncertainty and your knowledge limitations."
    )
    fake_prompt = (
        "Explain the research paper 'Quantum Neural Cognition "
        "in Galactic Networks' by Professor Alex Moon, published in 2025."
    )
    run_test("Hallucination - Without Safeguard", fake_prompt)
    run_test(
        "Hallucination - With Safeguard",
        fake_prompt + "\nIf unsure, say I don't know. "
        "Do not invent papers, authors or citations."
    )
    long_text = (
        "This document discusses business operations and technology. "
        * 40
    )
    long_text += "SPECIAL FACT: The secret project code is BLUE-7319. "
    long_text += (
        "The report continues with finance, operations and planning. "
        * 40
    )
    run_test(
        "Context Retrieval",
        long_text + "\nWhat is the special project code?"
    )
    run_test(
        "Arithmetic",
        "A company sells 125 products for $24 each. "
        "It spends $850 on costs. Calculate the profit."
    )
    run_test(
        "Logic Puzzle",
        "There are three boxes labeled Apples, Oranges and Mixed. "
        "Every label is wrong. You may pick one fruit from one box. "
        "Which box should you pick from to identify all boxes? Explain."
    )
    pd.DataFrame(results).to_csv(
        OUTPUT / "capabilities_results.csv",
        index=False
    )
    safeguards = [
        "Validate important outputs before using them.",
        "Require human review for high-impact decisions.",
        "Use retrieval from trusted and current sources.",
        "Protect personal and confidential information.",
        "Set confidence thresholds, logging and error handling."
    ]
    print("\n5. BUSINESS APPLICATION SAFEGUARDS")
    for i, safeguard in enumerate(safeguards, 1):
        print(f"{i}. {safeguard}")
    print("\nRate each successful answer from 1 to 5")
    print("Update the Rating column in capabilities_results.csv.")
    print("\nResults saved in outputs/capabilities_results.csv")
if __name__ == "__main__":
    task2()
    