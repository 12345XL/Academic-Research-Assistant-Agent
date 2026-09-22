"""Check project Markdown for delimiter mistakes in plain previewers."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def check(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors = []
    fence = None
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip(" ")
        if len(line) - len(stripped) <= 3 and stripped.startswith(("```", "~~~")):
            marker = stripped[0]
            length = len(stripped) - len(stripped.lstrip(marker))
            if fence is None:
                fence = (marker, length, line_number)
            elif marker == fence[0] and length >= fence[1] and not stripped[length:].strip():
                fence = None
        elif fence is None and "$" in line:
            errors.append(f"{path}:{line_number}: dollar sign may be parsed as math")
        if any(ord(char) < 32 and char not in "\t\r\n" for char in line):
            errors.append(f"{path}:{line_number}: unexpected control character")
    if fence is not None:
        errors.append(f"{path}:{fence[2]}: unclosed code fence")
    return errors


def main() -> int:
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.md"], cwd=ROOT)
    paths = sorted({ROOT / name.decode() for name in names.split(b"\0") if name})
    errors = [error for path in paths for error in check(path)]
    print("\n".join(errors) if errors else f"Markdown check passed: {len(paths)} files")
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
