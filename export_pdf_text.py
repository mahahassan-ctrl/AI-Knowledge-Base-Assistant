
from pathlib import Path
from pypdf import PdfReader

data_dir = Path("data")
output_path = Path("pdf_full_text.txt")

with output_path.open("w", encoding="utf-8") as output:
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))

        output.write(f"\n{'=' * 80}\n")
        output.write(f"DOCUMENT: {pdf_path.name}\n")
        output.write(f"TOTAL PAGES: {len(reader.pages)}\n")
        output.write(f"{'=' * 80}\n")

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            output.write(
                f"\n--- DOCUMENT: {pdf_path.name} | PAGE: {page_number} ---\n"
            )
            output.write(text.strip() + "\n")

print(f"Full PDF text exported to: {output_path}")
print(f"File size: {output_path.stat().st_size:,} bytes")
