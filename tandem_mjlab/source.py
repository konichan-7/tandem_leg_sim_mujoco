from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def source_files() -> list[Path]:
    return sorted(
        [*PACKAGE_DIR.glob("*.py"), *PACKAGE_DIR.glob("*.json")]
        + [
            path
            for path in (PACKAGE_DIR / "function").rglob("*")
            if path.is_file() and path.suffix in (".py", ".json", ".md")
        ]
        + [
            PACKAGE_DIR / "pyproject.toml",
            PACKAGE_DIR / "README.md",
            PACKAGE_DIR / "AGENTS.md",
        ]
    )
