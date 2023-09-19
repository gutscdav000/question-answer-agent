# software I had to install with brew, but do not want.
# popplar, tesseract
from pdf2image import convert_from_path
from PIL import Image
#from pytesseract import image_to_string
import pytesseract
import os, re



def list_files_in_directory(directory_path):
    """
    Lists all files from a given directory.

    Parameters:
    - directory_path: Path to the directory.

    Returns:
    - A list of file names.
    """
    return [ os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, filename))]

def write_to_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content        


# load PDF's, convert to images, read text from images, write raw text to files
def ingest_scripts():
    BASE_DIR = 'scripts-pdfs'
    FILE_NAME = 'the-fast-and-the-furious-2001.pdf'
    FILE_PATH = f'{BASE_DIR}/{FILE_NAME}'
    IMAGE_FOLDER = 'pdf-images/'

    for file_path in list_files_in_directory(BASE_DIR):
        file_name = file_path.split('/')[-1]
        # Convert the PDF to a list of image objects
        images = convert_from_path(file_path)
        text = [pytesseract.image_to_string(image) for image in images]
        print("-" * 20, end = '')
        print('text', end='')
        print("-" * 20)
        print(text)
        write_to_file(f'raw_{file_name}', ' '.join(text))

# cleanup files
FILE_NAME = 'the-fast-and-the-furious-2001.pdf'
BASE_DIR = 'raw-text'
text = read_file(f'{BASE_DIR}/raw_{FILE_NAME}')

int_ext_text_mod = text.replace("CONT'D", 'CONTINUED').replace("CONT ' D", 'CONTINUED').replace('INT', 'INTERIOR').replace('EXT', 'EXTERIOR').replace('POV', 'POINT OF VIEW')
# remove blue revision text
drop_blue_rev = re.sub(r"^.*Blue Revision.*\n?", '', int_ext_text_mod, flags=re.MULTILINE)
non_alphanumeric_chars = ''.join([char for char in drop_blue_rev if char.isalnum() or char in ',.;:\'"[]()-_!@#$%^&*' or char in ' \n'])
#print(''.join(non_alphanumeric_chars))
#print(set(non_alphanumeric_chars))
#remove_special_chars = re.sub(r'[^a-zA-Z0-9?!.,():;" ]', ' ', non_alphanumeric_chars)
remove_special_chars = non_alphanumeric_chars
new_file_name = FILE_NAME.split('.')[0]
write_to_file(f'the-fast-and-the-furious-2001/rev1_{new_file_name}', remove_special_chars)
        

    

