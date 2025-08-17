import os
import sys
import argparse
import re
import time
from PIL import Image

from gemini_vision import Gemini2Vision

class ImageFileRenamer:
    """
    A class to rename image files based on their content using a Gemini Vision model.
    Can process a single image file or recursively process a directory.
    """

    def __init__(self, gemini_model_name: str = Gemini2Vision.MODELS[0]):
        """
        Initializes the ImageFileRenamer with the Gemini model.

        Args:
            gemini_model_name (str): The name of the Gemini Vision model to use.
        """
        self.gemini = Gemini2Vision(gemini_model_name)
        self.supported_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a string to be a valid filename."""
        s = re.sub(r'[\\/:*?"<>|]', '', filename)
        s = s.replace(' ', '_')
        return s.strip()

    def create_resized_image(self, image_path: str, size: tuple = (512, 512)) -> str or None:
        """
        Creates a temporary, resized version of the image for description purposes.
        Args:
            image_path (str): The path to the original image file.
            size (tuple): The target size for the resized image.
        Returns:
            str: The path to the temporary resized image file, or None if an error occurs.
        """
        try:
            with Image.open(image_path) as img:
                temp_filename = f"{os.path.basename(image_path)}.tmp"
                temp_path = os.path.join(os.path.dirname(image_path), temp_filename)
                img.thumbnail(size, Image.Resampling.LANCZOS)
                img.save(temp_path, img.format)
                print(f"Created temporary resized image at: {temp_path}")
                return temp_path
        except Exception as e:
            print(f"Error resizing image {image_path}: {e}")
            return None

    def _process_single_image(self, original_filepath: str):
        """
        Processes and renames a single image file.
        """
        filename = os.path.basename(original_filepath)
        resized_filepath = None

        try:
            print(f"Processing {original_filepath}")

            if not filename.lower().endswith(self.supported_extensions):
                print(f"Skipping '{filename}': Not a supported image file.")
                return

            resized_filepath = self.create_resized_image(original_filepath)
            if not resized_filepath:
                print(f"Skipping {filename} due to resize error.")
                return

            img_data = self.gemini.load_image(resized_filepath)
            prompt = "Describe this image in 1-5 words, suitable for a filename. Do not include file extensions."
            description = self.gemini.get_response(prompt, img_data)

            _, file_extension = os.path.splitext(filename)
            sanitized_description = self.sanitize_filename(description)

            new_filename = f"{sanitized_description}{file_extension}"
            new_filepath = os.path.join(os.path.dirname(original_filepath), new_filename)

            counter = 1
            while os.path.exists(new_filepath):
                new_filename = f"{sanitized_description}_{counter}{file_extension}"
                new_filepath = os.path.join(os.path.dirname(original_filepath), new_filename)
                counter += 1

            os.rename(original_filepath, new_filepath)
            print(f"Renamed '{filename}' to '{new_filename}'")
            time.sleep(2) # Delay to avoid rate-limiting

        except Exception as e:
            print(f"Could not process '{filename}': {e}")
            time.sleep(2)
        finally:
            if resized_filepath and os.path.exists(resized_filepath):
                os.remove(resized_filepath)
                print(f"Cleaned up temporary file: {resized_filepath}")

    def rename_input(self, input_path: str):
        """
        Processes the input path, which can be a single image or a directory.
        If it's a directory, it recursively processes all image files within it.
        """
        if os.path.isfile(input_path):
            print(f"Processing single image: {input_path}")
            self._process_single_image(input_path)
        elif os.path.isdir(input_path):
            print(f"Processing directory recursively: {input_path}")
            for root, _, files in os.walk(input_path):
                for file in files:
                    filepath = os.path.join(root, file)
                    self._process_single_image(filepath)
        else:
            raise FileNotFoundError(f"Error: Input path '{input_path}' not found or is not a file/directory.")

def main():
    parser = argparse.ArgumentParser(description="Rename image files based on their content using Google Generative AI.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to a single image file or a directory containing images.")
    args = parser.parse_args()

    try:
        renamer = ImageFileRenamer()
        renamer.rename_input(args.input)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
