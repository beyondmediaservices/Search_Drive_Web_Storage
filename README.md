# Search_Drive_Web_Storage

how Retriever works in Gen AI:
Let’s say a user asks: “What’s the latest research on quantum computing?”

1) Query Embedding: The retriever will convert this question into an embedding that captures its meaning.
2) Search: The retriever searches a database or corpus of research papers, articles, or other documents for content related to “quantum computing research.”
3) Similarity: It finds the most relevant research papers or abstracts by measuring the semantic similarity to the query.
4) Response Generation: The generative model combines information from these documents to generate a concise, informative response.


for running GCSDirectoryLoader
1) pip install langchain google-cloud-storage pytesseract pdf2image

       If you have scanned PDFs or images, pytesseract (Tesseract OCR) is required. Make sure Tesseract is installed, and its path is set.
       We also need to download NLTK module "punkt"
       Run the following in Python:

       import nltk
       nltk.download('punkt')

3) brew install tesseract

To run the script:-
       python test.py -s GCS/GDRIVE
