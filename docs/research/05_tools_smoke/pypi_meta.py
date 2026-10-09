"""Print latest PyPI version, licence, Python requirement, release date and macOS-arm64 wheel
availability for each package named on the command line. Read-only (PyPI JSON API)."""
import json, sys, urllib.request

for name in sys.argv[1:]:
    try:
        d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=20))
    except Exception as e:
        print(f"{name:14s} ERROR {e}")
        continue
    i = d["info"]
    files = d["urls"]
    date = files[0]["upload_time"][:10] if files else "?"
    wheels = [f["filename"] for f in files if f["filename"].endswith(".whl")]
    arm = any(("macosx" in w and ("arm64" in w or "universal2" in w)) for w in wheels)
    pure = any(w.endswith("none-any.whl") for w in wheels)
    lic = i.get("license_expression") or (i.get("license") or "").split("\n")[0][:40]
    cls = [c.split("::")[-1].strip() for c in i.get("classifiers", []) if c.startswith("License")]
    print(f"{name:14s} {i['version']:12s} {date}  req={i.get('requires_python')!s:10s} "
          f"arm64_wheel={arm!s:5s} pure={pure!s:5s} lic={lic!r} cls={cls}")
