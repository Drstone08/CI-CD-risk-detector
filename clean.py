"""
clean_commits_csv_v2.py
------------------------
Cleans messy commit data and removes rows with broken or oversized patch text.

Output:
    commits_data_clean.csv
"""

import pandas as pd
import re

# 1️⃣ Load the raw data safely
df = pd.read_csv("commits_data_all.csv", on_bad_lines="skip", dtype=str, engine="python")

# 2️⃣ Drop rows missing key identifiers
df = df.dropna(subset=["repository", "commit_sha", "filename"], how="any")

# 3️⃣ Strip whitespace
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# 4️⃣ Clean patch text
def clean_patch(patch):
    if not isinstance(patch, str):
        return ""
    patch = patch.replace("\r", " ").replace("\n", " ").replace(",", " ")
    patch = re.sub(r"@@.*?@@", " ", patch)  # remove diff markers
    patch = re.sub(r"\s+", " ", patch)
    return patch.strip()

if "patch" in df.columns:
    df["patch"] = df["patch"].apply(clean_patch)
else:
    df["patch"] = ""

# 5️⃣ Remove rows with missing or invalid patch
def is_valid_patch(p):
    """Keep only reasonable, code-like patches."""
    if not isinstance(p, str) or len(p.strip()) == 0:
        return False
    if len(p) > 2500:  # too large → probably broken multi-row
        return False
    # Require at least some typical code/diff characters
    if not re.search(r"[+={};()\[\]]", p):
        return False
    return True

df = df[df["patch"].apply(is_valid_patch)]

# 6️⃣ Clean message and timestamp
if "message" in df.columns:
    df["message"] = df["message"].astype(str)
    df["message"] = df["message"].apply(lambda x: x.replace("\n", " ").replace(",", " "))

def clean_time(t):
    if not isinstance(t, str):
        return ""
    t = t.strip().replace("T", " ").replace("Z", "")
    try:
        return pd.to_datetime(t, errors="coerce").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return t

if "timestamp" in df.columns:
    df["timestamp"] = df["timestamp"].apply(clean_time)

# 7️⃣ Remove duplicates
df = df.drop_duplicates(subset=["repository", "commit_sha", "filename"])

# 8️⃣ Reorder columns
columns_order = [
    "repository", "commit_sha", "author", "timestamp",
    "message", "filename", "additions", "deletions", "changes", "patch"
]
df = df[[c for c in columns_order if c in df.columns]]

# 9️⃣ Save cleaned data
df.to_csv("commits_data_clean.csv", index=False)
print(f"✅ Cleaned data saved to commits_data_clean.csv with {len(df)} rows.")
