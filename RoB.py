from pathlib import Path
import base64
import PyPDF2
import openai
import json

class PDFtoROBAssessor:
    """Complete pipeline: PDF → Text → DeepSeek API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def extract_pdf_text(self, pdf_path: Path, use_ocr: bool = False) -> str:
        """Extract text from PDF"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num+1} ---\n{page_text}\n"
                    
            # If text is minimal, try OCR (optional)
            if use_ocr and len(text.strip()) < 500:
                text += self._extract_with_ocr(pdf_path)
                
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {e}")
            
        return text
    
    def pdf_to_base64_for_storage(self, pdf_path: Path) -> dict:
        """
        Convert PDF and extracted text to base64 for storage
        (NOT for sending to API)
        """
        # Extract text
        text = self.extract_pdf_text(pdf_path)
        
        # Encode for storage
        with open(pdf_path, "rb") as f:
            pdf_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        text_base64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
        
        return {
            "pdf_filename": pdf_path.name,
            "pdf_base64": pdf_base64,  # Store original PDF
            "extracted_text": text,  # For API calls
            "extracted_text_base64": text_base64  # For compressed storage
        }
    
    def assess_pdf(self, pdf_path: Path, author: str, year: str, reg_num: str) -> dict:
        """
        Main method: Extract text from PDF and send to DeepSeek API
        """
        print(f"Processing PDF: {pdf_path.name}")
        
        # 1. Extract text
        trial_text = self.extract_pdf_text(pdf_path)
        print(f"Extracted {len(trial_text)} characters")
        with open("t_text.txt", "w") as f:
            f.write(trial_text)
        
        # 2. Load guidelines
        with open("guidelines.txt", 'r', encoding='utf-8') as f:
            guidelines = f.read()
        
        # 3. Create prompt with EXTRACTED TEXT (not base64)
        prompt = f"""{guidelines}

TRIAL TO ASSESS:
First Author: {author}
Year: {year}
Registration Number: {reg_num}
Source PDF: {pdf_path.name}

EXTRACTED TRIAL TEXT:
{trial_text}

Now assess this trial. Provide output in JSON format."""
        
        # 4. Call API with TEXT
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # 5. Add metadata
        result["_source"] = {
            "pdf_file": str(pdf_path),
            "author": author,
            "year": year,
            "registration": reg_num,
            "text_length": len(trial_text)
        }
        
        return result

# Usage
assessor = PDFtoROBAssessor(api_key="sk-f62d916743074a7db6f6cdc8ce3e6e4b")
result = assessor.assess_pdf(
    pdf_path=Path("main.pdf"),
    author="Smith",
    year="2023",
    reg_num="NCT12345678"
)

# Save result
with open("rob_assessment.json", "w") as f:
    json.dump(result, f, indent=2)