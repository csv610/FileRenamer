import google.generativeai as genai
import os
import sys
import argparse
from PIL import Image

class Gemini2Vision:
    MODELS = ['gemini-2.5-flash', 
              'gemini-2.5-flash-lite'
             ]

    def __init__(self, model_name='gemini-2.5-flash'):
        self.model_name = model_name
        self.model = None
        self.configure_api()
        self.initialize_model()
    
    def configure_api(self):
        """Configure the Generative AI API."""
        google_api_key = os.getenv('GEMINI_API_KEY')
        if not google_api_key:
            print("Error: GEMINI_API_KEY environment variable not set.")
            sys.exit(1)
        genai.configure(api_key=google_api_key)
    
    def list_models(self):
        """List available Generative AI models."""
        print("Available Models:")
        for m in genai.list_models():
            print(m.name)
    
    def initialize_model(self):
        """Initialize the Generative AI model."""
        self.model = genai.GenerativeModel(self.model_name)
    
    @staticmethod
    def load_image(filename):
        """Load an image from a given filename."""
        try:
            return Image.open(filename)
        except Exception as e:
            print(f"Error: Unable to open image file {filename}. {e}")
            sys.exit(1)
    
    def get_response(self, prompt, img):
        """Generate content using the model, prompt, and image."""
        try:
            response = self.model.generate_content([prompt, img])
            return response.text
        except Exception as e:
            print(f"Error during content generation: {e}")
            sys.exit(1)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate content using Google Generative AI with Gemini model.")
    parser.add_argument("-i", "--filename", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("-q", "--question", type=str, default="Ask question about the image", help="Prompt text for content generation.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    prompt = args.question

    gemini = Gemini2Vision( Gemini2Vision.MODELS[0] )
    
    img = gemini.load_image(args.filename)
    text = gemini.get_response(prompt, img)
    
    print("Generated Content:")
    print(text)

if __name__ == "__main__":
    main()

