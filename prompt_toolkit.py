import json
import os
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from gemini_helper import ask_gemini, extract_json
BASE = Path(__file__).parent
DATA = BASE / "data"
PROMPTS = BASE / "prompts"
OUTPUT = BASE / "outputs"
OUTPUT.mkdir(exist_ok=True)
RUN_EXTENDED = (os.getenv("RUN_EXTENDED_EXPERIMENTS", "false").lower() == "true")
class Resume(BaseModel):
    name: str
    skills: list[str]
    years_experience: int = Field(ge=0)
def load_json(filename):
    with open(DATA / filename, "r", encoding="utf-8") as file:
        return json.load(file)
def load_prompt(filename):
    with open(PROMPTS / filename, "r", encoding="utf-8") as file:
        return file.read()
def save_json(filename, data):
    with open(OUTPUT / filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
def call_model(prompt, temperature=0, top_p=0.95, json_mode=False):
    answer, error = ask_gemini(
        prompt,
        temperature=temperature,
        top_p=top_p,
        json_mode=json_mode
    )
    if error:
        print("API unavailable:", error)
        return None
    return answer
def classify_ticket(prompt):
    answer = call_model(prompt, json_mode=True)
    if not answer:
        return None
    try:
        result = extract_json(answer)
        if result.get("category") not in ["Billing", "Technical", "Account"]:
            return None
        if result.get("priority") not in ["High", "Medium", "Low"]:
            return None
        return result
    except (ValueError, TypeError, AttributeError):
        return None
    
def classification_experiment():
    print("\nCLASSIFICATION ACCURACY")
    tickets = load_json("golden_tickets.json")
    versions = [
        "classification_v1.txt",
        "classification_v2.txt",
        "classification_v3.txt"
    ]
    all_results = []
    accuracy_results = {}
    for version in versions:
        template = load_prompt(version)
        correct = 0
        completed = 0
        print(f"\nTesting {version}")
        for ticket in tickets:
            prompt = template.replace(
                "{text}", ticket["text"]
            )
            prediction = classify_ticket(prompt)
            is_correct = False
            if prediction is not None:
                completed += 1
                is_correct = (
                    prediction["category"] == ticket["category"]
                    and prediction["priority"] == ticket["priority"]
                )
                if is_correct:
                    correct += 1
            all_results.append({
                "version": version,
                "ticket_id": ticket["id"],
                "ticket": ticket["text"],
                "expected_category": ticket["category"],
                "expected_priority": ticket["priority"],
                "predicted": prediction,
                "correct": is_correct
            })
        accuracy =correct / len(tickets) * 100
        accuracy_results[version] = {
            "correct": correct,
            "total": len(tickets),
            "accuracy": round(accuracy, 2),
            "completed": completed
        }
        print(f"{version}: {correct}/{len(tickets)} "f"({accuracy:.2f}%)")
    save_json("classification_details.json", all_results)
    save_json("classification_accuracy.json", accuracy_results)
    return accuracy_results
def reasoning_experiment():
    print("\nWORD PROBLEM EXPERIMENT")
    problems = [
        {
            "question": "A shop sells 12 pens for $5 each. What is the total?",
            "answer": "60"
        },
        {
            "question": "A car travels 180 km in 3 hours. What is its speed?",
            "answer": "60"
        },
        {
            "question": "A product costs $800 and has a 10% discount. What is its final price?",
            "answer": "720"
        },
        {
            "question": "There are 25 students. 40% are girls. How many girls?",
            "answer": "10"
        },
        {
            "question": "A person has 100 apples and sells 35. How many remain?",
            "answer": "65"
        }
    ]
    results = []
    for problem in problems:
        for style in ["Plain", "Step-by-step prompt"]:
            if style == "Plain":
                prompt =("Solve this problem and give the final answer: "+ problem["question"])
            else:
                prompt = (
                    "Work through this problem step by step internally. "
                    "Give a concise explanation and the final answer: "
                    + problem["question"]
                )
            answer = call_model(prompt)
            results.append({
                "problem": problem["question"],
                "style": style,
                "expected_answer": problem["answer"],
                "model_answer": answer,
                "verified_by_hand": problem["answer"]
            })
            print("\nProblem:", problem["question"])
            print("Style:", style)
            print("Expected:", problem["answer"])
            print("Model:", answer)
    save_json("reasoning_results.json", results)
def creative_experiment():
    print("\nCREATIVE GENERATION EXPERIMENT")
    prompt = (
        "Write a short creative story about a robot "
        "that discovers a hidden garden."
    )
    results = []
    for temperature in [0, 0.7, 1.2]:
        for repetition in range(1, 6):
            answer = call_model(
                prompt,
                temperature=temperature,
                top_p=0.95
            )
            results.append({
                "temperature": temperature,
                "top_p": 0.95,
                "repetition": repetition,
                "answer": answer
            })
            print(f"\nTemperature {temperature}, "f"Run {repetition}")
            print(answer)
    for top_p in [0.3, 0.95]:
        for repetition in range(1, 6):
            answer = call_model(prompt,temperature=0.7,top_p=top_p)
            results.append({
                "temperature": 0.7,
                "top_p": top_p,
                "repetition": repetition,
                "answer": answer
            })
            print(f"\nTop-p {top_p}, Run {repetition}")
            print(answer)
    save_json("creative_results.json", results)
def resume_experiment():
    print("\nRESUME EXTRACTION")
    resumes = load_json("resumes.json")
    results = []
    for resume in resumes:
        text = resume["text"]
        expected = resume["expected"]
        prompt = f"""
        Extract the following information from the resume.
        Resume: {text}
        Return JSON with:
        name, skills, years_experience.
        Do not guess missing information.
        """
        plain_answer = call_model(prompt,temperature=0)
        plain_valid = False
        plain_data = None
        if plain_answer:
            try:
                plain_data = extract_json(plain_answer)
                Resume.model_validate(plain_data)
                plain_valid = True
            except (ValueError, ValidationError, TypeError):
                pass
        json_answer = call_model(
            prompt,
            temperature=0,
            json_mode=True
        )
        json_valid = False
        json_data = None
        if json_answer:
            try:
                json_data = extract_json(json_answer)
                Resume.model_validate(json_data)
                json_valid = True
            except (ValueError, ValidationError, TypeError):
                pass
        results.append({
            "resume_id": resume["id"],
            "expected": expected,
            "plain_answer": plain_answer,
            "plain_valid": plain_valid,
            "json_answer": json_answer,
            "json_valid": json_valid
        })
        print("\nResume:", resume["id"])
        print("Expected:", expected)
        print("Plain JSON valid:", plain_valid)
        print("JSON mode valid:", json_valid)
    plain_invalid = sum(
        not item["plain_valid"] for item in results
    )
    json_invalid = sum(
        not item["json_valid"] for item in results
    )
    print("\nInvalid JSON Summary")
    print("Plain mode invalid:", plain_invalid, "/", len(resumes))
    print("JSON mode invalid:", json_invalid, "/", len(resumes))
    save_json("resume_results.json", results)
    return {
        "plain_invalid": plain_invalid,
        "json_invalid": json_invalid
    }
def prompt_improvement_experiment():
    print("\nBAD PROMPTS AND IMPROVEMENTS")
    examples = [
        {
            "bad": "Classify this.",
            "improved": (
                "Classify the following customer ticket as Billing, "
                "Technical or Account. Assign High, Medium or Low priority. "
                "Return only valid JSON with category and priority. "
                "Ticket: I cannot log in to my account."
            )
        },
        {
            "bad": "Extract details.",
            "improved": (
                "Extract name, skills and years of experience from this "
                "resume. Use an integer for experience and a list for skills. "
                "Return valid JSON. Do not invent missing details. "
                "Resume: Anita is a Python developer with 3 years "
                "of experience in SQL and Python."
            )
        },
        {
            "bad": "Summarize this text.",
            "improved": (
                "Summarize the following article in exactly three bullet "
                "points. Preserve important facts and do not introduce "
                "new information. Text: Artificial Intelligence helps "
                "businesses automate tasks and analyze data."
            )
        }
    ]
    results = []
    for item in examples:
        print("\nBad Prompt:", item["bad"])
        bad_result = call_model(item["bad"])
        print("Bad Prompt Result:", bad_result)
        print("\nImproved Prompt:", item["improved"])
        improved_result = call_model(item["improved"])
        print("Improved Result:", improved_result)
        results.append({
            "bad_prompt": item["bad"],
            "bad_result": bad_result,
            "improved_prompt": item["improved"],
            "improved_result": improved_result
        })
    save_json("prompt_improvements.json", results)
def task4():
    print("\nPROMPT ENGINEERING TOOLKIT")
    accuracy = classification_experiment()
    print("\nClassification Accuracy:")
    print(json.dumps(accuracy, indent=4))
    if RUN_EXTENDED:
        reasoning_experiment()
        creative_experiment()
        resume_experiment()
        prompt_improvement_experiment()
    else:
        print("\nExtended experiments were skipped.")
        print("Set RUN_EXTENDED_EXPERIMENTS=true in .env")
        print("to run reasoning, temperature, top-p,")
        print("resume extraction and prompt improvements.")

if __name__ == "__main__":
    task4()
