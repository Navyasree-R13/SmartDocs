
from pypdf import PdfReader

pdf_path = r"C:\Users\Welcome\Downloads\AES AAT.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    extracted_text = page.extract_text()
    if extracted_text:
        text += extracted_text + "\n"

print("PDF TEXT EXTRACTED SUCCESSFULLY!")
print("--------------------------------")
print(text)
