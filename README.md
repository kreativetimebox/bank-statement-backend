Document OCR and Named Entity Recognition (NER)
This project uses deep learning models to perform Optical Character Recognition (OCR) on an image and then runs Named Entity Recognition (NER) on the extracted text. It processes a document image line-by-line, identifies text, and then extracts entities like names of people, organizations, and locations from that text.
The final, structured output is saved as a unique JSON file.
Features
High-Performance OCR: Utilizes the doctr library to accurately detect and recognize text from an image.
Named Entity Recognition: Employs a pre-trained transformers model from Hugging Face (dslim/bert-base-NER) to identify entities.
Structured JSON Output: Saves the results in a clean, human-readable JSON format.
Unique Filenames: Automatically generates a unique filename for each run based on the input image's name and a timestamp, preventing overwrites.
Requirements
This project requires Python 3.8+ and the following libraries:
python-doctr (with a PyTorch backend)
transformers
torch (PyTorch)
scipy
