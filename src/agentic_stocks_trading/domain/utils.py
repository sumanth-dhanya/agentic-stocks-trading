from pathlib import Path


def find_project_root() -> Path:
    """Find the project root directory by looking for README.md"""
    current_dir = Path.cwd()

    # Start from the current directory and go up the tree
    while current_dir != current_dir.parent:
        if (current_dir / "README.md").exists() or (current_dir / "Dockerfile").exists():
            return current_dir
        current_dir = current_dir.parent

    # If no README.md is found, use the current directory as fallback
    return Path.cwd()
