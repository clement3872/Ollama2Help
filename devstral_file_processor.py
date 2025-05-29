#!/usr/bin/env python3
"""
Devstral File Processor - A tool that processes multiple files (text/PDF) 
and uses Ollama's Devstral model to answer questions based on file contents.

Devstral is optimized for agentic coding tasks, so this tool structures
prompts to leverage its methodical problem-solving capabilities.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import requests
import PyPDF2
from dataclasses import dataclass

@dataclass
class FileContent:
    """Represents processed file content with metadata"""
    path: str
    content: str
    file_type: str
    size: int
    encoding: Optional[str] = None
    pages: Optional[int] = None

class DevstralFileProcessor:
    """
    File processor optimized for Devstral's agentic capabilities.
    Devstral excels at methodical exploration and analysis.
    """
    
    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "devstral"):
        self.ollama_host = ollama_host.rstrip('/')
        self.model = model
        self.files_processed: List[FileContent] = []
        
    def read_text_file(self, file_path: str) -> FileContent:
        """Read and process text files with encoding detection"""
        path = Path(file_path)
        
        # Try different encodings
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        content = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()
                    used_encoding = encoding
                    break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise ValueError(f"Could not decode file {file_path} with any supported encoding")
        
        return FileContent(
            path=str(path.absolute()),
            content=content,
            file_type="text",
            size=len(content),
            encoding=used_encoding
        )
    
    def read_pdf_file(self, file_path: str) -> FileContent:
        """Read and extract text from PDF files"""
        path = Path(file_path)
        content_parts = []
        
        try:
            with open(path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        content_parts.append(f"--- Page {page_num + 1} ---\n{text}")
                
                content = "\n\n".join(content_parts)
                
        except Exception as e:
            raise ValueError(f"Error reading PDF {file_path}: {str(e)}")
        
        return FileContent(
            path=str(path.absolute()),
            content=content,
            file_type="pdf",
            size=len(content),
            pages=num_pages
        )
    
    def read_file(self, file_path: str) -> FileContent:
        """Automatically detect file type and read accordingly"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        file_extension = path.suffix.lower()
        
        if file_extension == '.pdf':
            return self.read_pdf_file(file_path)
        else:
            return self.read_text_file(file_path)
    
    def process_files(self, file_paths: List[str]) -> None:
        """Process multiple files and store their contents"""
        self.files_processed = []
        
        print(f"Processing {len(file_paths)} files...")
        
        for file_path in file_paths:
            try:
                file_content = self.read_file(file_path)
                self.files_processed.append(file_content)
                print(f"✓ Processed {file_path} ({file_content.file_type}, {file_content.size} chars)")
            except Exception as e:
                print(f"✗ Error processing {file_path}: {e}")
                continue
    
    def create_devstral_prompt(self, user_question: str) -> str:
        """
        Create a structured prompt optimized for Devstral's agentic capabilities.
        Devstral performs best with methodical, exploration-focused prompts.
        """
        
        # Build file context section
        file_context = []
        for i, file_content in enumerate(self.files_processed, 1):
            context = f"""
FILE {i}: {file_content.path}
Type: {file_content.file_type.upper()}
Size: {file_content.size} characters
"""
            if file_content.encoding:
                context += f"Encoding: {file_content.encoding}\n"
            if file_content.pages:
                context += f"Pages: {file_content.pages}\n"
            
            context += f"""
Content:
{'-' * 60}
{file_content.content}
{'-' * 60}
"""
            file_context.append(context)
        
        # Structure the prompt to leverage Devstral's methodical approach
        prompt = f"""<ROLE>
You are Devstral, an agentic model specialized in thorough analysis and problem-solving. You have been provided with {len(self.files_processed)} file(s) to analyze and a question to answer.
</ROLE>

<TASK>
Analyze the provided files and answer the following question:
{user_question}
</TASK>

<METHODOLOGY>
Please approach this systematically:
1. EXPLORATION: First, thoroughly explore and understand the content of each file
2. ANALYSIS: Identify relevant information that relates to the question
3. SYNTHESIS: Combine insights from multiple files if applicable
4. RESPONSE: Provide a comprehensive, well-structured answer
</METHODOLOGY>

<FILE_CONTEXT>
{''.join(file_context)}
</FILE_CONTEXT>

<INSTRUCTIONS>
- Be thorough and methodical in your analysis
- Reference specific files and sections when relevant
- If the files don't contain enough information to fully answer the question, clearly state what's missing
- Provide actionable insights when possible
- Focus on quality and accuracy over speed
</INSTRUCTIONS>

Please proceed with your analysis and provide your response."""

        return prompt
    
    def query_ollama(self, prompt: str) -> Dict:
        """Send prompt to Ollama and get response"""
        url = f"{self.ollama_host}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Lower temperature for more focused responses
                "top_p": 0.9,
                "num_predict": 4096  # Allow longer responses for thorough analysis
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")
    
    def ask_question(self, question: str) -> str:
        """Process question using Devstral with file context"""
        if not self.files_processed:
            raise ValueError("No files have been processed. Please load files first.")
        
        print("Generating structured prompt for Devstral...")
        prompt = self.create_devstral_prompt(question)
        
        print(f"Querying {self.model} model...")
        response = self.query_ollama(prompt)
        
        if 'response' not in response:
            raise ValueError("Invalid response from Ollama")
        
        return response['response']
    
    def get_processing_summary(self) -> Dict:
        """Get summary of processed files"""
        return {
            "total_files": len(self.files_processed),
            "files": [
                {
                    "path": f.path,
                    "type": f.file_type,
                    "size": f.size,
                    "encoding": f.encoding,
                    "pages": f.pages
                }
                for f in self.files_processed
            ],
            "total_content_size": sum(f.size for f in self.files_processed)
        }

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Process files and answer questions using Ollama's Devstral model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python devstral_processor.py file1.txt file2.pdf --question "What are the main themes?"
  python devstral_processor.py *.py --question "Find potential bugs in this code"
  python devstral_processor.py report.pdf --host http://192.168.1.100:11434
        """
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        help='Paths to files to process (text files and PDFs)'
    )
    
    parser.add_argument(
        '--question', '-q',
        required=True,
        help='Question to ask about the files'
    )
    
    parser.add_argument(
        '--host',
        default='http://localhost:11434',
        help='Ollama host URL (default: http://localhost:11434)'
    )
    
    parser.add_argument(
        '--model', '-m',
        default='devstral',
        help='Ollama model to use (default: devstral)'
    )
    
    parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help='Show processing summary'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = DevstralFileProcessor(
            ollama_host=args.host,
            model=args.model
        )
        
        # Process files
        processor.process_files(args.files)
        
        if not processor.files_processed:
            print("No files were successfully processed. Exiting.")
            sys.exit(1)
        
        # Show summary if requested
        if args.summary:
            summary = processor.get_processing_summary()
            print(f"\nProcessing Summary:")
            print(f"Files processed: {summary['total_files']}")
            print(f"Total content size: {summary['total_content_size']} characters")
            print()
        
        # Ask question
        print(f"\nQuestion: {args.question}")
        print("\n" + "="*60)
        
        answer = processor.ask_question(args.question)
        print(answer)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
