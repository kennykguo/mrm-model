"""Assemble docs/tlc/tlc_g26_evaluation.html from template.html and data.json (see extract.py)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "tlc_g26_evaluation.html"


def main() -> None:
    data = (HERE / "data.json").read_text().replace("</", "<\\/")      # keep the JSON inside its <script>
    out = (HERE / "template.html").read_text().replace("__DATA__", data)
    OUT.write_text(out)
    print(OUT, len(out) // 1024, "KiB")


if __name__ == "__main__":
    main()
