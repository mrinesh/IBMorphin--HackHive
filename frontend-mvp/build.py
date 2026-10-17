"""Builds index.html (the page published for review) from src/template.html and
src/engine3d-lite.js. The page needs no backend: all data is generated in the browser."""
from pathlib import Path

root = Path(__file__).parent
page = (root / "src/template.html").read_text(encoding="utf-8")
engine = (root / "src/engine3d-lite.js").read_text(encoding="utf-8")
assert "/*ENGINE_JS*/" in page
(root / "index.html").write_text(page.replace("/*ENGINE_JS*/", engine), encoding="utf-8")
print("index.html written", len(page) + len(engine), "bytes")
