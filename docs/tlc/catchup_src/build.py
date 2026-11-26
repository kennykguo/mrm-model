"""Assemble docs/tlc/catchup.html from template.html, timeline.html, learn.html and data.json (see extract.py)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    t = (HERE / "template.html").read_text()
    data = (HERE / "data.json").read_text().replace("</", "<\\/")      # keep the JSON inside its <script>
    out = (t.replace("__TIMELINE__", (HERE / "timeline.html").read_text())
            .replace("__LEARN__", (HERE / "learn.html").read_text()).replace("__DATA__", data))
    (HERE.parent / "catchup.html").write_text(out)
    print(HERE.parent / "catchup.html", len(out) // 1024, "KiB")


if __name__ == "__main__":
    main()
