import os
import subprocess
import shutil
import tempfile

# File extensions to include
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".cpp", ".c", ".h", ".cs",
    ".go", ".rs", ".rb", ".php",
    ".md", ".txt", ".json", ".yaml", ".yml", ".env.example"
}

# Folders to skip
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv",
    "venv", "dist", "build", ".next", ".cache"
}


def load_github(repo_url: str) -> list[dict]:
    """
    Clone a GitHub repo and extract text from all relevant source files.
    Returns a list of dicts with file path and content.
    """
    tmp_dir = tempfile.mkdtemp()

    try:
        print(f"[GitHub] Cloning '{repo_url}'...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, tmp_dir],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Git clone failed: {result.stderr}")

        files = []

        for root, dirs, filenames in os.walk(tmp_dir):
            # remove skipped directories in-place
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in ALLOWED_EXTENSIONS:
                    continue

                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, tmp_dir)

                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                    if content:
                        files.append({
                            "source": repo_url,
                            "file": relative_path,
                            "content": content
                        })
                except Exception as e:
                    print(f"[GitHub] Skipping {relative_path}: {e}")

        print(f"[GitHub] Loaded {len(files)} files from '{repo_url}'")
        return files

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------- quick test ----------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python github_loader.py <github_repo_url>")
        sys.exit(1)

    files = load_github(sys.argv[1])
    for f in files[:3]:  # preview first 3 files
        print(f"\n--- {f['file']} ---")
        print(f["content"][:300])