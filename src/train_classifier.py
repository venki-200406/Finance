"""Train and compare transaction classifiers, tracking every run in MLflow.

Run from the project root: python -m src.train_classifier
View the runs with: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import joblib
import mlflow
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.predict import MODEL_PATH, ROOT, clean_text

DATA_PATH = ROOT / "data" / "transactions.csv"
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT = "transaction-classifier"
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Character n-grams can still match merchant names that contain typos.
VECTORIZERS = {
    "word": {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2},
    "char": {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2},
}
CLASSIFIERS = {
    "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    "linear_svc": LinearSVC(class_weight="balanced", random_state=SEED),
}


def load_split(path=DATA_PATH):
    """Return the fixed (X_train, X_test, y_train, y_test) split of cleaned descriptions."""
    if not path.exists():
        raise SystemExit(f"No data at {path}. Run: python -m src.generate_data")
    df = pd.read_csv(path)
    return train_test_split(
        df["description"].map(clean_text),
        df["category"],
        test_size=TEST_SIZE,
        stratify=df["category"],
        random_state=SEED,
    )


def build_pipeline(vectorizer: str, classifier: str) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**VECTORIZERS[vectorizer])),
            ("clf", clone(CLASSIFIERS[classifier])),
        ]
    )


def run_experiment(vectorizer: str, classifier: str, X_train, X_test, y_train, y_test) -> dict:
    """Fit one configuration, log it to MLflow and return its results."""
    pipeline = build_pipeline(vectorizer, classifier)
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    with mlflow.start_run(run_name=f"{classifier}_{vectorizer}") as run:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=folds, scoring="f1_macro")
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        labels = sorted(y_test.unique())
        per_class = f1_score(y_test, predictions, average=None, labels=labels)
        mlflow.log_params(
            {
                "classifier": classifier,
                "vectorizer": vectorizer,
                "C": CLASSIFIERS[classifier].C,
                "class_weight": "balanced",
                **VECTORIZERS[vectorizer],
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }
        )
        mlflow.log_metrics(
            {
                "cv_macro_f1": float(cv_scores.mean()),
                "cv_macro_f1_std": float(cv_scores.std()),
                "test_macro_f1": f1_score(y_test, predictions, average="macro"),
                **{f"f1_{label}": float(score) for label, score in zip(labels, per_class)},
            }
        )

    return {
        "name": f"{classifier}_{vectorizer}",
        "run_id": run.info.run_id,
        "cv_macro_f1": float(cv_scores.mean()),
        "test_macro_f1": f1_score(y_test, predictions, average="macro"),
        "pipeline": pipeline,
        "predictions": predictions,
    }


def main() -> None:
    X_train, X_test, y_train, y_test = load_split()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    results = [
        run_experiment(vectorizer, classifier, X_train, X_test, y_train, y_test)
        for classifier in CLASSIFIERS
        for vectorizer in VECTORIZERS
    ]

    # Choosing on cross-validation keeps the test score an honest estimate.
    best = max(results, key=lambda result: result["cv_macro_f1"])
    mlflow.MlflowClient().set_tag(best["run_id"], "best_model", "true")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], MODEL_PATH, compress=3)

    print(f"{'run':<20}{'cv_macro_f1':>13}{'test_macro_f1':>15}")
    for result in results:
        print(f"{result['name']:<20}{result['cv_macro_f1']:>13.3f}{result['test_macro_f1']:>15.3f}")
    print(f"\nBest: {best['name']} (saved to {MODEL_PATH})\n")
    print(classification_report(y_test, best["predictions"], digits=3))


if __name__ == "__main__":
    main()
