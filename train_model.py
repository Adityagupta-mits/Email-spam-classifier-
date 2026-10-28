"""
train_model.py
--------------
Trains multiple ML models on the SMS Spam Collection dataset,
compares them, and saves the best model + TF-IDF vectorizer.

Usage:
    python train_model.py

Output:
    model/spam_model.pkl
    model/tfidf_vectorizer.pkl
    model/model_metrics.json
"""

import os
import json
import warnings
import urllib.request
import zipfile
import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from preprocess import clean_text, ensure_nltk_data

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_PATH = os.path.join(DATASET_DIR, "emails.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────
FALLBACK_DATA = {
    "label": (
        ["ham"] * 70 + ["spam"] * 70
    ),
    "text": (
        # Ham samples
        [
            "Hey, are you free this weekend?",
            "Can you send me the report by Friday?",
            "Meeting rescheduled to 3pm tomorrow.",
            "Happy birthday! Hope you have a great day.",
            "Dinner at 7? Let me know if that works.",
            "Call me when you get a chance.",
            "The project is going well, thanks for asking.",
            "I'll be late to the office today.",
            "Can you review my code changes?",
            "Let's catch up over coffee next week.",
            "Thanks for the help yesterday!",
            "Please find the attached document.",
            "I need your input on the budget proposal.",
            "The meeting has been postponed to next Monday.",
            "How was your vacation? Hope it was great!",
            "Could you please review the slides before the presentation?",
            "I've completed the task you assigned.",
            "Let me know if you need anything else.",
            "The team lunch is at 1pm today.",
            "Your feedback on the design would be helpful.",
            "I will send you the files shortly.",
            "Are you attending the conference next week?",
            "The server maintenance is scheduled for tonight.",
            "Please RSVP to the company event by Friday.",
            "Good morning! How are you doing?",
            "I wanted to follow up on yesterday's discussion.",
            "The client approved the proposal, great news!",
            "Can we schedule a quick call this afternoon?",
            "Please update the database with the new entries.",
            "Looking forward to seeing you at the event.",
            "Your subscription has been renewed successfully.",
            "The delivery is expected by Thursday.",
            "I'll be working from home tomorrow.",
            "Thanks for attending the webinar.",
            "The test results look good.",
            "Please submit your timesheet by end of day.",
            "The new policy has been implemented.",
            "Your request has been approved.",
            "The system update is complete.",
            "Please confirm your availability for the meeting.",
            "I received your message and will respond shortly.",
            "The product demo went very well.",
            "Happy to help with anything you need.",
            "The quarterly report is ready for review.",
            "Your account password has been changed as requested.",
            "The training session starts at 9am tomorrow.",
            "Let me know if you have any questions.",
            "The invoice has been processed.",
            "Your flight is confirmed for next Tuesday.",
            "The library resources are available online.",
            "Can you share the agenda for tomorrow's meeting?",
            "I'll pick you up at 6pm.",
            "The new office opens next month.",
            "Please review the contract before signing.",
            "Your package has been shipped.",
            "The annual report is now available.",
            "Please update your contact information.",
            "The event has been cancelled due to weather.",
            "I'll send you the login credentials shortly.",
            "Your feedback is important to us.",
            "The team has completed the migration.",
            "Please find your ticket attached.",
            "The lecture notes are available on the portal.",
            "Your appointment is confirmed for Monday.",
            "The budget meeting is at 2pm.",
            "Thanks for your patience.",
            "The application form is attached.",
            "Your enrollment is confirmed.",
            "Please submit the assignment by midnight.",
            "The exam results will be out next week.",
        ],
        # Spam samples
        [
            "Congratulations! You won a $1,000,000 lottery. Claim now!",
            "FREE iPhone 15! Click here to collect your prize today!",
            "URGENT: Your bank account has been suspended. Verify now.",
            "Win big! Earn $500 daily from home. Limited slots available!",
            "You have been selected for an exclusive reward. Click here.",
            "Act now! Free gift card worth $200. No strings attached!",
            "Your account is at risk. Click immediately to secure it.",
            "Earn money online. 100% guaranteed. Sign up today!",
            "WINNER! You've won a free vacation. Claim your prize now.",
            "URGENT ACTION REQUIRED: Suspicious activity on your account.",
            "Get rich fast! Invest now and earn millions.",
            "Congratulations, you've been selected for a special offer!",
            "Free money! Click here to get $1000 instantly.",
            "Your credit card has been compromised. Verify your details.",
            "WIN a brand new car! Enter now, it's FREE!",
            "Make $5000 per week from home. Guaranteed income.",
            "ALERT: Your PayPal account needs verification. Click now.",
            "Exclusive offer for you! Discount up to 90% off.",
            "Free lottery ticket! You could win $10 million.",
            "Click here to claim your exclusive bonus.",
            "You owe taxes. Pay immediately to avoid arrest.",
            "FREE GIFT! Just pay shipping. Limited time offer.",
            "Bitcoin investment opportunity. 300% returns guaranteed.",
            "Congratulations on being our 1 millionth visitor!",
            "Your subscription expires. Renew now for free.",
            "Hot singles in your area! Click to view profiles.",
            "Lose 30 pounds in 30 days! Doctors hate this trick.",
            "Your Amazon account is locked. Verify now.",
            "Earn unlimited cash from home. Zero investment needed.",
            "Final warning: Update your payment to avoid suspension.",
            "Free iPhone! Take our survey and claim your prize.",
            "You won a gift card! Click here to redeem.",
            "Make money fast with our secret system.",
            "Your loan has been approved! Claim $10,000 now.",
            "Cheap medication online. No prescription needed.",
            "Click here to unsubscribe or your account will be charged.",
            "WINNER: You are selected for a $500 Amazon voucher.",
            "Work from home and earn $3000 weekly. Start today!",
            "Your email has been flagged. Verify to continue.",
            "FREE casino bonus! Gamble and win millions!",
            "Limited time: 80% off on all products. Shop now!",
            "Get your free credit report instantly.",
            "Refinance your mortgage. Save thousands now.",
            "Double your money in 24 hours. Guaranteed.",
            "Your inheritance is waiting. Contact us immediately.",
            "FINAL NOTICE: You owe $2,500. Pay now to avoid legal action.",
            "Unlock premium features for free. Click here.",
            "Earn commissions by referring friends. No investment.",
            "Your Netflix account will expire. Update billing now.",
            "Free trial available. Sign up with no credit card required.",
            "CLAIM your reward before it expires in 24 hours.",
            "Celebrity diet secret revealed. Lose weight fast!",
            "Investment tip: This stock will triple in value!",
            "Win an all-expenses paid trip. Enter now for free!",
            "Your bank requires verification. Click the link below.",
            "SALE ENDS TONIGHT! 70% off everything.",
            "Congratulations! Your number was selected. Reply now.",
            "Join millions making money online. 100% legitimate.",
            "GET PAID to take surveys from home. $50/hour.",
            "Your account will be deleted unless you log in now.",
            "Lowest rates on insurance. Get a free quote today.",
            "You have an unclaimed refund of $890. Claim now.",
            "OFFICIAL NOTICE: Claim your government stimulus payment.",
            "Earn daily income. Fully automated. Zero effort.",
            "Hot deal! Buy one, get one FREE today only.",
            "Phishing: Confirm your password to restore access.",
            "Fr€€ M0n€y! Click h3re to $ecure your pr1ze now.",
            "Verify your account or it will be permanently closed.",
            "You've been selected for an IRS audit. Resolve now.",
            "Sign up for our pyramid scheme and earn passively!",
        ]
    )
}


def download_dataset():
    """Try to download the SMS Spam Collection dataset."""
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = os.path.join(DATASET_DIR, "sms_spam.zip")
    raw_path = os.path.join(DATASET_DIR, "SMSSpamCollection")

    print("  Downloading SMS Spam Collection dataset...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATASET_DIR)
        os.remove(zip_path)

        # Parse tab-separated file
        df = pd.read_csv(raw_path, sep="\t", header=None, names=["label", "text"],
                         encoding="latin-1")
        df = df[["label", "text"]].dropna()
        df.to_csv(DATASET_PATH, index=False)
        print(f"  Dataset saved: {len(df)} emails ({df['label'].value_counts().to_dict()})")
        return df

    except Exception as e:
        print(f"  Download failed ({e}). Using built-in fallback dataset.")
        return None


def load_dataset():
    """Load dataset from disk, download if missing, fall back to built-in sample."""
    if os.path.exists(DATASET_PATH):
        print("  Loading existing dataset...")
        df = pd.read_csv(DATASET_PATH)
        print(f"  Loaded {len(df)} records.")
        return df

    df = download_dataset()
    if df is None:
        print("  Using fallback dataset (140 emails).")
        df = pd.DataFrame(FALLBACK_DATA)
        df.to_csv(DATASET_PATH, index=False)
    return df


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train_and_evaluate():
    ensure_nltk_data()
    print("\n" + "=" * 60)
    print("  AI EMAIL SPAM CLASSIFIER - MODEL TRAINING")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading dataset...")
    df = load_dataset()
    df["label_enc"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label_enc"])

    # Preprocess
    print("[2/5] Preprocessing text (this may take a moment)...")
    df["clean"] = df["text"].apply(clean_text)

    # TF-IDF vectorisation
    print("[3/5] Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    X = vectorizer.fit_transform(df["clean"])
    y = df["label_enc"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Models
    print("[4/5] Training all models...")
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Support Vector Machine": LinearSVC(max_iter=1000, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    results = {}
    best_model = None
    best_name = ""
    best_f1 = 0.0

    print("\n" + "-" * 60)
    print(f"  {'Model':<30} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "confusion_matrix": cm,
        }

        print(f"  {name:<30} {acc*100:>6.2f}% {prec*100:>6.2f}% {rec*100:>6.2f}% {f1*100:>6.2f}%")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    print("-" * 60)
    print(f"\n  * Best model: {best_name} (F1={best_f1*100:.2f}%)")

    # Label distribution
    dist = df["label"].value_counts().to_dict()

    # Save artefacts
    print("\n[5/5] Saving model and vectorizer...")
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    metrics_payload = {
        "best_model": best_name,
        "dataset_size": len(df),
        "class_distribution": dist,
        "model_results": results,
        "features": vectorizer.get_feature_names_out()[:20].tolist(),
        "top_spam_features": _get_top_spam_features(best_name, best_model, vectorizer),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"  Model  -> {MODEL_PATH}")
    print(f"  TF-IDF -> {VECTORIZER_PATH}")
    print(f"  Metrics-> {METRICS_PATH}")
    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60 + "\n")

    return best_model, vectorizer, metrics_payload


def _get_top_spam_features(model_name: str, model, vectorizer) -> list:
    """Extract top spam-class features from supported models."""
    try:
        feature_names = vectorizer.get_feature_names_out()
        if hasattr(model, "feature_log_prob_"):
            # Naive Bayes
            spam_log_probs = model.feature_log_prob_[1]
            top_indices = np.argsort(spam_log_probs)[-20:][::-1]
            return feature_names[top_indices].tolist()
        elif hasattr(model, "coef_"):
            # LR / SVM
            coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            top_indices = np.argsort(coef)[-20:][::-1]
            return feature_names[top_indices].tolist()
    except Exception:
        pass
    return []


if __name__ == "__main__":
    train_and_evaluate()
