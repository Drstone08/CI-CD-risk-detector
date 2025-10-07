"""
collect_commits_multi.py
-------------------------
Collects recent commits and file diffs from multiple GitHub repositories,
and saves a clean, Excel-friendly dataset (no line breaks, no commas in patch).

Output:
    commits_data_all.csv — combined dataset with commits from all repos.
"""

import requests
import pandas as pd
import time
import re

# ====================================
# CONFIGURATION
# ====================================

# ⚠️ Use your personal GitHub token (read-only)
# Create one at: https://github.com/settings/tokens
TOKEN = "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"

# 👉 List of repositories to process (owner/repo format)
REPOSITORIES = [
    "vercel/next.js",
    "facebook/react",
    "actions/checkout",
    "kubernetes/kubernetes",
]

# Number of commits per repo
NUM_COMMITS_PER_REPO = 50

HEADERS = {"Authorization": f"token {TOKEN}"}

# ====================================
# HELPER FUNCTIONS
# ====================================

def clean_patch_text(patch_text):
    """
    Cleans raw GitHub patch text:
    - Removes line breaks, commas, and @@ diff markers
    - Keeps only safe, single-line text
    """
    if not isinstance(patch_text, str):
        return ""
    patch_text = patch_text.replace("\n", " ").replace("\r", " ").replace(",", " ")
    patch_text = re.sub(r"@@.*?@@", " ", patch_text)
    patch_text = re.sub(r"\s+", " ", patch_text)
    return patch_text.strip()


def fetch_commits(repo, num_commits):
    """Fetch basic commit metadata (sha, author, date, message)."""
    print(f"\n📦 Fetching commits from {repo} ...")
    commits = []
    page = 1
    per_page = 30
    while len(commits) < num_commits:
        url = f"https://api.github.com/repos/{repo}/commits?per_page={per_page}&page={page}"
        r = requests.get(url, headers=HEADERS)
        if r.status_code != 200:
            print(f"❌ Error {r.status_code} for {repo}: {r.text}")
            break
        data = r.json()
        if not data:
            break
        commits.extend(data)
        page += 1
        time.sleep(0.5)
    return commits[:num_commits]


def fetch_commit_details(repo, sha):
    """Fetch detailed info (files, additions, deletions, patches) for one commit."""
    url = f"https://api.github.com/repos/{repo}/commits/{sha}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        print(f"❌ Error fetching {sha[:7]} from {repo}: {r.status_code}")
        return None
    return r.json()


def collect_repo_data(repo, num_commits):
    """Collect commits and file details for one repo."""
    commits = fetch_commits(repo, num_commits)
    rows = []
    for commit in commits:
        sha = commit["sha"]
        details = fetch_commit_details(repo, sha)
        if not details:
            continue

        author = details["commit"]["author"]["name"]
        timestamp = details["commit"]["author"]["date"]
        message = details["commit"]["message"].replace("\n", " ").replace(",", " ")
        files = details.get("files", [])

        for f in files:
            patch = clean_patch_text(f.get("patch", ""))
            rows.append({
                "repository": repo,
                "commit_sha": sha,
                "author": author,
                "timestamp": timestamp,
                "message": message,
                "filename": f["filename"],
                "additions": f["additions"],
                "deletions": f["deletions"],
                "changes": f["changes"],
                "patch": patch
            })
        time.sleep(0.3)
    print(f"✅ {repo}: collected {len(rows)} file-level changes.")
    return rows


def main():
    all_rows = []

    for repo in REPOSITORIES:
        repo_rows = collect_repo_data(repo, NUM_COMMITS_PER_REPO)
        all_rows.extend(repo_rows)
        time.sleep(1)  # small pause between repos

    df = pd.DataFrame(all_rows)

    # Remove duplicates or broken rows
    df = df.dropna(subset=["repository", "commit_sha", "filename"])
    df = df.drop_duplicates(subset=["repository", "commit_sha", "filename"])

    # Save cleaned dataset
    df.to_csv("commits_data_all.csv", index=False)
    print("\n🎉 All done!")
    print(f"Total clean records collected: {len(df)}")
    print("Saved to commits_data_all.csv")


if __name__ == "__main__":
    main()
