# data_gen.py
"""
Generate a synthetic medical triage dataset for training and EDA.
"""

import random
import csv

SYMPTOMS = [
    "Fever", "Cough", "Shortness of breath", "Chest pain",
    "Headache", "Fatigue", "Sore throat", "Nausea",
    "Vomiting", "Diarrhea", "Abdominal pain",
    "Dizziness", "Loss of smell", "Loss of taste"
]

RED_FLAGS = [
    "Severe bleeding", "Unconscious", "Slurred speech", "Fainting", "Severe chest pain"
]


def generate_record():
    age = random.randint(1, 90)
    duration_days = random.choice([1, 2, 3, 4, 5, 7, 10, 14])

    num_sym = random.randint(1, 4)
    symptoms = random.sample(SYMPTOMS, num_sym)

    red_flags = []
    # small chance for red flag
    if random.random() < 0.15:
        red_flags = [random.choice(RED_FLAGS)]

    return {
        "age": age,
        "duration_days": duration_days,
        "symptoms": ";".join(symptoms),
        "red_flags": ";".join(red_flags)
    }


def generate_csv(path="synthetic_vignettes.csv", n=600):
    fieldnames = ["age", "duration_days", "symptoms", "red_flags"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(n):
            writer.writerow(generate_record())
    print(f"Generated {n} synthetic rows at {path}")


if __name__ == "__main__":
    generate_csv()
