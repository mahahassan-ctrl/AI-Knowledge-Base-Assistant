
from pathlib import Path
from pypdf import PdfReader

DATA_DIR = Path("data")

for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
    reader = PdfReader(str(pdf_path))

    print("\n" + "=" * 90)
    print(f"DOCUMENT: {pdf_path.name}")
    print(f"TOTAL PAGES: {len(reader.pages)}")
    print("=" * 90)

    for page_number, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())

        if text:
            preview = text[:350]
            print(f"\n[Page {page_number}] {preview}")
        else:
            print(f"\n[Page {page_number}] No extractable text")
