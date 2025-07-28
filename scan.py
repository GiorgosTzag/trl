#!/usr/bin/env python3
"""
AI Passport - Repository Dataset Scanner (cleaned + licence/risk enrichment)

- Scans repository for dataset references
- Skips our own files / obvious placeholders
- Enriches each reference with a best-guess licence + risk note
- Writes .passport/summary.json and .passport/summary.md
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.parse import urlparse
import socket

HTTP_TIMEOUT = 8  # seconds

class DatasetScanner:
    def __init__(self, root_dir="."):
        self.root_dir = Path(root_dir)
        self.passport_dir = self.root_dir / ".passport"
        self.datasets_found = []

    # ---------- SCAN ----------
    def scan_repository(self):
        """Scan the repository for dataset references."""
        print("🔍 Scanning repository for dataset references...")

        # regex patterns
        patterns = {
            "huggingface_load_dataset": r"load_dataset\s*\(\s*['\"]([^'\"]+)['\"]",
            "pandas_read_csv": r"(?:pd\.read_csv|pandas\.read_csv)\s*\(\s*['\"]([^'\"]+)['\"]",
            "dataset_urls": r"['\"]([^'\"]*(?:\.csv|\.jsonl|\.parquet|\.zip))['\"]",
            # groups: (optional schema), (org/name or path after /datasets/)
            "huggingface_datasets": r"['\"]?(https?://)?(?:www\.)?huggingface\.co/datasets/([^'\"\\s]+)['\"]?"
        }

        file_extensions = {'.py', '.ipynb', '.md', '.txt', '.yml', '.yaml', '.json'}
        # folders to skip
        skip_dirs = {
            ".git", ".passport", "node_modules", "venv", "env", "dist",
            "build", ".ipynb_checkpoints", "data"
        }
        # files to skip (to avoid matching our own examples)
        skip_files = {
            "scan.py", "README.md", ".github/workflows/passport.yml"
        }

        for file_path in self.root_dir.rglob("*"):
            if file_path.is_dir():
                continue
            if file_path.suffix.lower() not in file_extensions:
                continue
            if any(part in skip_dirs or part.startswith('.') for part in file_path.parts):
                continue

            relative_path = file_path.relative_to(self.root_dir)
            if str(relative_path).replace("\\", "/") in skip_files:
                continue

            try:
                self._scan_file(file_path, relative_path, patterns)
            except Exception as e:
                print(f"⚠️  Error scanning {file_path}: {e}")

        print(f"✅ Found {len(self.datasets_found)} dataset references")
        return self.datasets_found

    def _scan_file(self, file_path: Path, relative_path: Path, patterns: dict):
        text = ""
        try:
            text = open(file_path, 'r', encoding='utf-8', errors='ignore').read()
        except Exception:
            return

        GENERIC_TOKENS = {
            "name", "dataset_name", ".csv", ".jsonl", ".parquet", ".zip"
        }

        for pattern_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                # normalise match by pattern
                if isinstance(m, tuple):
                    # For huggingface_datasets we want the 2nd group (org/name)
                    if pattern_type == "huggingface_datasets":
                        m = m[1]
                    else:
                        m = ''.join(m)
                ref = (m or "").strip()
                if not ref or ref in GENERIC_TOKENS or len(ref) < 3:
                    continue

                entry = {
                    "type": pattern_type,
                    "reference": ref,
                    "file": str(relative_path).replace("\\", "/"),
                    "found_at": datetime.now().isoformat(),
                }

                # de-duplicate by type+reference (case-insensitive)
                key = f"{entry['type']}:{entry['reference'].lower()}"
                if any(f"{d['type']}:{d['reference'].lower()}" == key for d in self.datasets_found):
                    continue
                self.datasets_found.append(entry)

    # ---------- ENRICH ----------
    def enrich_with_licences(self):
        """Add licence / risk info to each dataset reference."""
        print("🧠 Classifying licences & risk...")
        for d in self.datasets_found:
            licence, source, notes, risk = self._classify_reference(d)
            d["licence"] = licence
            d["licence_source"] = source
            d["notes"] = notes
            d["risk_flag"] = risk

    def _classify_reference(self, d):
        """Return (licence, source, notes, risk_flag) for a dataset entry."""
        ref = d["reference"]
        ptype = d["type"]

        # 1) Hugging Face: explicit URL or load_dataset("name")
        hf_id = None
        if ptype == "huggingface_datasets":
            hf_id = ref
        elif ptype == "huggingface_load_dataset":
            hf_id = ref

        if hf_id:
            # normalise id (strip protocol/host if present)
            hf_id = hf_id.replace("https://huggingface.co/datasets/", "").strip().strip("/")
            lic = self._fetch_hf_license(hf_id)
            if lic:
                return (lic, "huggingface_api", f"License read from HF card for '{hf_id}'", self._risk_from_licence(lic))
            else:
                return ("Unknown", "heuristic", f"No license field found for '{hf_id}'", "review")

        # 2) Direct URLs to data files
        if self._looks_like_url(ref):
            host = urlparse(ref).netloc.lower()
            if "data.gov" in host or "europa.eu" in host:
                return ("Open Government", "heuristic", f"Public sector domain '{host}'", "low")
            if "kaggle.com" in host:
                return ("Varies (Kaggle)", "heuristic", "Check the dataset page license on Kaggle", "review")
            if "huggingface.co" in host:
                return ("Unknown", "heuristic", "HF link but not a dataset card; inspect manually", "review")
            return ("Unknown", "heuristic", f"External host '{host}'", "review")

        # 3) Local files (CSV/JSONL/Parquet) or relative paths
        if any(ref.lower().endswith(ext) for ext in [".csv", ".jsonl", ".parquet", ".zip"]):
            return ("Proprietary/Project-internal", "heuristic", "Local file path in repo", "review")

        # fallback
        return ("Unknown", "heuristic", "Pattern matched but no license inference", "review")

    def _fetch_hf_license(self, hf_id):
        """Try to read dataset license from Hugging Face public API."""
        url = f"https://huggingface.co/api/datasets/{hf_id}"
        try:
            req = Request(url, headers={"User-Agent": "ai-passport-scanner"})
            with urlopen(req, timeout=HTTP_TIMEOUT) as r:
                data = json.loads(r.read().decode("utf-8", errors="ignore"))
            card = data.get("cardData") or {}
            lic = card.get("license")
            if not lic:
                ls = card.get("licenses")
                if isinstance(ls, list) and ls:
                    if isinstance(ls[0], dict) and "name" in ls[0]:
                        lic = ls[0]["name"]
                    elif isinstance(ls[0], str):
                        lic = ls[0]
            if lic:
                return str(lic)
        except Exception:
            pass
        return None

    def _looks_like_url(self, s: str) -> bool:
        return s.startswith("http://") or s.startswith("https://")

    def _risk_from_licence(self, lic: str) -> str:
        lic_l = (lic or "").lower()
        if any(k in lic_l for k in ["cc0", "public domain", "open government", "odc", "apache", "mit", "bsd"]):
            return "low"
        if "cc-by" in lic_l or "cc by" in lic_l:
            return "low"
        if lic_l in ("unknown", "", "none", "other"):
            return "review"
        return "review"

    # ---------- OUTPUT ----------
    def generate_passport(self):
        print("📝 Generating passport files...")
        self.passport_dir.mkdir(exist_ok=True)

        summary_data = {
            "scanned_at": datetime.now().isoformat(),
            "total_datasets": len(self.datasets_found),
            "datasets": self.datasets_found,
            "types": self._get_type_summary()
        }

        with open(self.passport_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        with open(self.passport_dir / "summary.md", "w", encoding="utf-8") as f:
            f.write(self._generate_markdown_summary(summary_data))

        print(f"✅ Generated passport files in {self.passport_dir}")
        return True

    def _get_type_summary(self):
        type_counts = {}
        for d in self.datasets_found:
            t = d["type"]
            type_counts[t] = type_counts.get(t, 0) + 1
        return type_counts

    def _generate_markdown_summary(self, summary_data):
        md = []
        md.append("# AI Passport - Dataset Summary\n")
        md.append(f"**Scanned at:** {summary_data['scanned_at']}  \n")
        md.append(f"**Total datasets found:** {summary_data['total_datasets']}\n\n")
        md.append("## Dataset Types\n\n")
        for dtype, count in summary_data['types'].items():
            md.append(f"- **{dtype.replace('_',' ').title()}:** {count}\n")

        if summary_data['datasets']:
            md.append("\n## Dataset References\n\n")
            for i, d in enumerate(summary_data["datasets"], 1):
                md.append(f"### {i}. {d['reference']}\n\n")
                md.append(f"- **Type:** {d['type'].replace('_',' ').title()}\n")
                md.append(f"- **File:** `{d['file']}`\n")
                lic = d.get("licence","Unknown")
                src = d.get("licence_source","")
                risk = d.get("risk_flag","review")
                notes = d.get("notes","")
                md.append(f"- **Licence:** {lic}  \n")
                md.append(f"- **Source:** {src}  \n")
                md.append(f"- **Risk:** {risk}  \n")
                if notes:
                    md.append(f"- **Notes:** {notes}\n")
                md.append("\n")
        else:
            md.append("\n## No datasets found\n\nNo dataset references were detected in this repository.\n")

        md.append("\n---\n*Generated by AI Passport Scanner*\n")
        return "".join(md)

def main():
    print("🚀 AI Passport Scanner starting...")
    scanner = DatasetScanner()
    scanner.scan_repository()
    scanner.enrich_with_licences()
    scanner.generate_passport()
    print("🎉 AI Passport scan completed!")

if __name__ == "__main__":
    socket.setdefaulttimeout(HTTP_TIMEOUT)
    main()
