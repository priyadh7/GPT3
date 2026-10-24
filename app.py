
from flask import Flask, render_template, request
from pathlib import Path
from datetime import datetime
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "emails.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "generated_reply.txt"
OUTPUT_FILE.parent.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

TONES = ["Friendly", "Formal", "Empathetic"]

INTENTS = [
    "Delivery Issue",
    "Refund Request",
    "Product Complaint",
    "Technical Support",
    "Order Cancellation",
    "General Inquiry"
]


def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def find_similar_email(email):
    empty = {
        "previous_email": "",
        "previous_reply": "",
        "similarity": 0.0
    }

    if not DATA_FILE.exists():
        return empty

    df = pd.read_csv(DATA_FILE).fillna("")

    if "Customer_Email" not in df.columns or df.empty:
        return empty

    emails = df["Customer_Email"].astype(str).tolist()
    valid_indices = [
        i for i, old_email in enumerate(emails)
        if old_email.strip()
    ]

    if not valid_indices:
        return empty

    valid_emails = [emails[i] for i in valid_indices]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(valid_emails + [email])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()

    best_position = int(scores.argmax())
    best_index = valid_indices[best_position]

    return {
        "previous_email": emails[best_index],
        "previous_reply": str(
            df.iloc[best_index].get("Previous_Reply", "")
        ),
        "similarity": round(float(scores[best_position]), 2)
    }


def detect_intent(subject, email):
    prompt = f"""
Classify the customer message into exactly one of these categories:
{", ".join(INTENTS)}

Subject: {subject}
Customer message: {email}

Return only one category name from the list. Do not explain.
"""
    answer = ask_ollama(prompt)

    for intent in INTENTS:
        if intent.lower() in answer.lower():
            return intent

    return "General Inquiry"


def generate_reply(name, subject, email, intent, tone,
                   previous_email, previous_reply):
    prompt = f"""
You are a professional customer support email assistant.

Customer name: {name}
Email subject: {subject}
Customer message: {email}
Detected intent: {intent}
Required tone: {tone}

Similar historical customer email:
{previous_email or "No similar email available"}

Previous response:
{previous_reply or "No previous response available"}

Write a complete email reply that:
- Addresses the customer by name.
- Responds directly to their concern.
- Uses a {tone.lower()} tone throughout.
- Is clear, helpful and professional.
- Uses the previous response as context only when relevant.
- Does not invent tracking details, completed actions, or refund approvals.
- Includes an appropriate greeting and sign-off.

Return only the email reply.
"""
    return ask_ollama(prompt)


def summarize_email(email):
    prompt = f"""
Summarize the customer's issue in no more than 50 words.
Include the main concern and any requested action.
Do not invent facts.

Customer message:
{email}

Return only the summary.
"""
    return ask_ollama(prompt)


def save_reply(name, subject, email, intent, tone, reply, summary):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        file.write(
            f"\nDate: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"Customer: {name}\n"
            f"Subject: {subject}\n"
            f"Customer Email: {email}\n"
            f"Intent: {intent}\n"
            f"Tone: {tone}\n"
            f"Generated Reply:\n{reply}\n"
            f"Summary: {summary}\n"
            + "-" * 60 + "\n"
        )


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        subject = request.form.get("subject", "").strip()
        email = request.form.get("email", "").strip()
        tone = request.form.get("tone", "Friendly")

        if not name or not subject or not email:
            error = "Please complete all the customer email fields."
        elif tone not in TONES:
            error = "Please select a valid tone."
        else:
            try:
                similar = find_similar_email(email)
                intent = detect_intent(subject, email)

                reply = generate_reply(
                    name, subject, email, intent, tone,
                    similar["previous_email"],
                    similar["previous_reply"]
                )

                summary = summarize_email(email)

                result = {
                    "name": name,
                    "subject": subject,
                    "email": email,
                    "tone": tone,
                    "intent": intent,
                    "reply": reply,
                    "summary": summary,
                    **similar
                }

                save_reply(
                    name, subject, email, intent, tone, reply, summary
                )

            except requests.RequestException:
                error = (
                    "Could not connect to Ollama. Ensure Ollama is "
                    "running and the llama3.2 model is installed."
                )
            except Exception as exc:
                app.logger.exception("Email assistant error")
                error = f"Unable to generate the reply: {exc}"

    return render_template(
        "index.html", result=result, error=error, tones=TONES
    )


if __name__ == "__main__":
    app.run(debug=True)
