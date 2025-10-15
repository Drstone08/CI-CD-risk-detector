import pandas as pd
import joblib
import requests
from pathlib import Path
from scipy import sparse
import os
from tabulate import tabulate  # <-- For pretty table formatting

def _extract_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature extraction (same as training)."""
    df2 = pd.DataFrame(index=df.index)
    df2["num_additions"] = pd.to_numeric(df["additions"], errors="coerce").fillna(0)
    df2["num_deletions"] = pd.to_numeric(df["deletions"], errors="coerce").fillna(0)
    df2["commit_hour"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.hour.fillna(-1)
    df2["is_ci_file"] = df["filename"].str.contains(r"\.github/workflows|ci\.yml", case=False, na=False).astype(int)
    df2["is_script_file"] = df["filename"].str.contains(r"\.(?:sh|py|yml|json|tf)$", case=False, na=False).astype(int)
    token_pat = r"(?i)(?:api[_-]?key|access[_-]?token|secret|password|AKIA[0-9A-Z]{16})"
    df2["has_token_like"] = df["patch"].str.contains(token_pat, regex=True, na=False).astype(int)
    return df2


class CommitRiskPredictor:
    """Loads a trained model and predicts the risk of a new commit."""

    def __init__(self, model_dir: str = "model_artifacts"):
        model_path = Path(model_dir)
        self.model = joblib.load(model_path / "best_model.pkl")
        self.vectorizer = joblib.load(model_path / "tfidf_vectorizer.joblib")
        self.scaler = joblib.load(model_path / "scaler.joblib")
        print("✅ Model and preprocessors loaded successfully.\n")

    def _fetch_commit_data(self, repo: str, sha: str) -> pd.DataFrame:
        """Fetches data for a single commit from GitHub API."""
        url = f"https://api.github.com/repos/{repo}/commits/{sha}"
        token = os.getenv("GITHUB_TOKEN")
        headers = {"Authorization": f"token {token}"} if token else {}

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"❌ Error fetching commit: {response.status_code} {response.text}")
            return pd.DataFrame()

        data = response.json()
        files = data.get("files", [])
        rows = []
        for f in files:
            rows.append({
                "timestamp": data.get("commit", {}).get("author", {}).get("date", ""),
                "filename": f.get("filename", ""),
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
                "patch": f.get("patch", "") or ""
            })
        return pd.DataFrame(rows)

    def predict(self, repo: str, sha: str, pretty_print: bool = True):
        """Predicts the risk for a given commit repo@sha and returns detailed info."""
        commit_df = self._fetch_commit_data(repo, sha)
        if commit_df.empty:
            return "❌ Could not fetch or process commit data."

        predictions = []
        for _, file_data in commit_df.iterrows():
            file_df = pd.DataFrame([file_data])

            X_struct = _extract_structured_features(file_df)
            X_struct_scaled = self.scaler.transform(X_struct)
            X_text = self.vectorizer.transform(file_df["patch"])
            X_all = sparse.hstack([X_text, sparse.csr_matrix(X_struct_scaled)], format="csr")

            risk_label = self.model.predict(X_all)[0]
            if hasattr(self.model, "predict_proba"):
                risk_proba = self.model.predict_proba(X_all)[0][1]
            else:
                # For models without predict_proba
                risk_proba = float(self.model.decision_function(X_all)[0])
                risk_proba = 1 / (1 + pow(2.71828, -risk_proba))

            predictions.append({
                "filename": file_data["filename"],
                "additions": int(file_data["additions"]),
                "deletions": int(file_data["deletions"]),
                "is_risky": "✅ YES" if risk_label == 1 else "❌ NO",
                "risk_score (%)": round(risk_proba * 100, 2)
            })

        df_results = pd.DataFrame(predictions)

        if pretty_print:
            print(f"\n🔍 Risk Analysis for commit {repo}@{sha[:7]}")
            print(tabulate(df_results, headers='keys', tablefmt='grid', showindex=False))
            overall_risk = df_results["risk_score (%)"].mean()
            print(f"\n📊 Average Commit Risk Score: {overall_risk:.2f}%\n")

        return df_results


# Example usage
if __name__ == '__main__':
    predictor = CommitRiskPredictor(model_dir="model_artifacts")

    # # Example commit (replace with your repo/sha)
    # repo_name = "vercel/next.js"
    # commit_sha = "4ee2f4ad8711401c51e51d0f941b1aeb803b537e"

    repo_name="withastro/astro"
    commit_sha="6ee63bf"

    results = predictor.predict(repo_name, commit_sha)
