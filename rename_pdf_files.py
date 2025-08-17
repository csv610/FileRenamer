import os
import sys
import PyPDF2
import google.generativeai as genai

class PDFRenamer:
    """
    A class to rename PDF files based on their title, using Gemini for intelligent
    title extraction when metadata is not available.
    """
    def __init__(self):
        """
        Initializes the Gemini API and model.
        """
        self.google_api_key = os.getenv('GEMINI_API_KEY')
        if not self.google_api_key:
            print("Error: The GEMINI_API_KEY environment variable is not set.")
            sys.exit(1)

        genai.configure(api_key=self.google_api_key)
        self.model_name = "gemini-2.5-flash"
        self.model = genai.GenerativeModel(self.model_name)

    def _get_pdf_metadata_title(self, pdf_path):
        """
        (Private method) Attempts to extract the title from the PDF's metadata.
        Returns the title as a string or None if not found.
        """
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                metadata = reader.metadata
                if metadata and '/Title' in metadata:
                    return metadata['/Title']
                return None
        except Exception as e:
            print(f"Error extracting metadata from {pdf_path}: {e}")
            return None

    def _get_first_page_text(self, pdf_path, num_pages=2):
        """
        (Private method) Extracts text from the first specified number of pages of a PDF.
        Returns the extracted text as a string.
        """
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text_content = ""
                for i in range(min(num_pages, len(reader.pages))):
                    text_content += reader.pages[i].extract_text() or ""
                return text_content
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def _get_title_from_gemini(self, text_content):
        """
        (Private method) Uses Gemini to intelligently extract the paper's title.
        Returns the title as a string or None if an error occurs.
        """
        if not text_content.strip():
            return None

        prompt_text = (
            "Analyze the following text from a document and extract the main title of the paper. "
            "Provide only the title, and do not include any other text, explanations, or formatting.\n\n"
            f"Document Content:\n{text_content[:2000]}"
        )
        try:
            response = self.model.generate_content(prompt_text)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            return None

    def _sanitize_filename(self, filename):
        """
        (Private method) Removes invalid characters from a string to make it a valid filename.
        """
        invalid_chars = '<>:"/\\|?*\n\r\t'
        for char in invalid_chars:
            filename = filename.replace(char, '')
        filename = '_'.join(filename.split()).strip()
        return filename

    def process_file(self, pdf_path):
        """
        Orchestrates the renaming process for a single PDF.
        It first tries metadata, then falls back to Gemini if needed.
        """
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path} ❌")
            return

        print(f"Processing: {pdf_path}")
        original_directory, original_filename = os.path.split(pdf_path)
        base_name, extension = os.path.splitext(original_filename)

        new_title = self._get_pdf_metadata_title(pdf_path)

        if not new_title:
            print("Metadata title not found. Falling back to Gemini...")
            text_content = self._get_first_page_text(pdf_path)
            if text_content:
                new_title = self._get_title_from_gemini(text_content)
            else:
                print(f"Could not extract text from {pdf_path}. Cannot rename. 🤷")
                return

        if new_title:
            sanitized_title = self._sanitize_filename(new_title)
            if not sanitized_title:
                print(f"Sanitized title is empty for {pdf_path}. Cannot rename. 🤷")
                return

            final_new_filename = f"{sanitized_title}{extension}"
            counter = 1
            while os.path.exists(os.path.join(original_directory, final_new_filename)):
                final_new_filename = f"{sanitized_title}_{counter}{extension}"
                counter += 1

            new_pdf_path = os.path.join(original_directory, final_new_filename)

            if pdf_path != new_pdf_path:
                try:
                    os.rename(pdf_path, new_pdf_path)
                    print(f"Renamed '{original_filename}' to '{final_new_filename}' ✅")
                except OSError as e:
                    print(f"Error renaming file '{original_filename}': {e} ❌")
            else:
                print(f"File '{original_filename}' already has an appropriate name. Skipping. 👍")
        else:
            print(f"Could not determine a suitable title for '{original_filename}'. 🤷")

    def run(self, input_path):
        """
        Handles the user input to determine if it's a file or a directory
        and processes the PDF files accordingly.
        """
        if os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
            self.process_file(input_path)
        elif os.path.isdir(input_path):
            print(f"Processing all PDF files in directory: {input_path}")
            for filename in os.listdir(input_path):
                if filename.lower().endswith('.pdf'):
                    full_path = os.path.join(input_path, filename)
                    self.process_file(full_path)
        else:
            print(f"Invalid input: '{input_path}' is not a valid PDF file or directory.")
            sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <path_to_pdf_file_or_directory>")
        sys.exit(1)

    renamer = PDFRenamer()
    input_path = sys.argv[1]
    renamer.run(input_path)

