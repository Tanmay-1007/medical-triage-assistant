# rules.py
"""
Simple rule-based triage engine.

Returns:
  category: one of ["Emergency", "Urgent", "Routine", "Self-care"]
  reasons: list[str] explanation
  probs: dict[str,float] pseudo-probabilities for categories
"""

CATEGORIES = ["Emergency", "Urgent", "Routine", "Self-care"]

def triage_rules(symptoms, red_flags, age, duration_days):
    symptoms = symptoms or []
    red_flags = red_flags or []
    age = age or 0
    duration_days = duration_days or 1

    reasons = []

    # 1) Hard red-flag emergencies
    emergency_flags = {"Severe bleeding", "Unconscious", "Slurred speech", "Severe chest pain"}
    if emergency_flags.intersection(set(red_flags)):
        reasons.append("Red-flag sign present → classify as Emergency.")
        return "Emergency", reasons, _probs("Emergency")

    # 2) Chest pain + breathlessness → Emergency
    if ("Chest pain" in symptoms and "Shortness of breath" in symptoms):
        reasons.append("Chest pain + shortness of breath → possible cardiac event.")
        return "Emergency", reasons, _probs("Emergency")

    # 3) Elderly with serious symptoms
    if age >= 65 and any(s in symptoms for s in ["Shortness of breath", "Chest pain", "Severe chest pain"]):
        reasons.append("Older age with severe symptom → Emergency.")
        return "Emergency", reasons, _probs("Emergency")

    # --- Urgent ---
    # Prolonged high fever or vomiting / diarrhea
    if ("Fever" in symptoms and duration_days >= 5) or \
       ("Vomiting" in symptoms and duration_days >= 3) or \
       ("Diarrhea" in symptoms and duration_days >= 3):
        reasons.append("Prolonged fever or GI symptoms → Urgent care needed.")
        return "Urgent", reasons, _probs("Urgent")

    # Moderate chest pain, dizziness, abdominal pain
    if any(s in symptoms for s in ["Chest pain", "Dizziness", "Abdominal pain"]) and duration_days >= 2:
        reasons.append("Persistent moderate symptom → Urgent category.")
        return "Urgent", reasons, _probs("Urgent")

    # --- Routine ---
    if any(s in symptoms for s in ["Fever", "Cough", "Sore throat", "Headache", "Fatigue"]):
        reasons.append("Common, non-severe symptoms → Routine review suggested.")
        return "Routine", reasons, _probs("Routine")

    # --- Self-care ---
    reasons.append("Mild or unspecified symptoms → Self-care and observation.")
    return "Self-care", reasons, _probs("Self-care")


def _probs(main_label):
    base = {c: 0.05 for c in CATEGORIES}
    base[main_label] = 0.8
    # normalize
    s = sum(base.values())
    return {k: v / s for k, v in base.items()}
