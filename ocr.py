import os
import time
import fitz  # PyMuPDF
import requests
import faiss
import numpy as np
import torch
import shutil
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai


def extract_images_from_pdf(pdf_path, output_folder="images"):
    """Extract images from a PDF and save them to a folder."""
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    doc = fitz.open(pdf_path)
    image_count = 0

    for page_number in range(len(doc)):
        for img_index, img in enumerate(doc[page_number].get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_filename = os.path.join(output_folder, f"{image_count}.{img_ext}")
            with open(img_filename, "wb") as img_file:
                img_file.write(image_bytes)
            image_count += 1
    print(f"Extracted {image_count} images from {pdf_path} into '{output_folder}' folder.")


def azure_text_detection(image_path, subscription_key, endpoint):
    """Extract text from an image using Azure OCR, skipping any errors."""
    try:
        ocr_url = f"{endpoint}/vision/v3.2/read/analyze"
        headers = {"Ocp-Apim-Subscription-Key": subscription_key, "Content-Type": "application/octet-stream"}

        with open(image_path, "rb") as image_file:
            response = requests.post(ocr_url, headers=headers, data=image_file)
            response.raise_for_status()

        operation_url = response.headers.get("Operation-Location")
        if not operation_url:
            return ""  # Skip if no operation URL

        while True:
            try:
                result = requests.get(operation_url, headers={"Ocp-Apim-Subscription-Key": subscription_key}).json()
                if result.get("status") == "succeeded":
                    break
                elif result.get("status") == "failed":
                    return ""  # Skip failed OCR attempts
                time.sleep(1)
            except requests.exceptions.RequestException:
                return ""  # Skip on request failure

        return " ".join(
            [line["text"] for line in result.get("analyzeResult", {}).get("readResults", [{}])[0].get("lines", [])])

    except (requests.exceptions.RequestException, KeyError, IndexError, Exception):
        return ""  # Skip on any error

def process_images_for_text(image_dir, output_file, subscription_key, endpoint):
    """Extract text from all images in a directory and save to a file."""
    with open(output_file, "w", encoding="utf-8") as file:
        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(image_dir, filename)
                extracted_text = azure_text_detection(image_path, subscription_key, endpoint)
                file.write(f"Image: {filename}\nExtracted Text:\n{extracted_text}\n\n")
                print(f"Processed: {filename}")
    print("Text extraction completed.")


def get_embedding(text, tokenizer, model):
    """Generate an embedding using Hugging Face transformers."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def create_faiss_index(text_chunks, model, tokenizer):
    """Create a FAISS index from text chunks."""
    embeddings = np.array([get_embedding(chunk, tokenizer, model) for chunk in text_chunks])
    # Initialize FAISS index
    dimension = embeddings.shape[1]  # Get embedding size dynamically
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, "faiss_index.bin")


def search_in_faiss(query, text_chunks, index, tokenizer, model, top_k=10):
    """Search FAISS index for the most relevant text chunks."""
    query_embedding = get_embedding(query, tokenizer, model).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return [text_chunks[i] for i in indices[0] if i < len(text_chunks)]


def generate_gemini_response(query, context, api_key):
    """Generate structured response using Free Gemini Model."""
    prompt = f"Based on the following data given, answer the question:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text if hasattr(response, 'text') else str(response)


def main():
    """Main function to run the entire pipeline."""
    pdf_path = "Innovision Event Instructions 2.pdf"
    image_dir = "images"
    output_file = "output1.txt"
    subscription_key = "BH5xbCyGRDbOEjhOXQuKZy2u5DMv2ioUQxjwaWx0pGhDkriwLJdLJQQJ99ALACGhslBXJ3w3AAAFACOGdWuH"
    endpoint = "https://chennai.cognitiveservices.azure.com/"  # e.g., "https://<your-resource-name>.cognitiveservices.azure.com/"
    api_key = "AIzaSyBp4rx0mN65nLJxAJIsphq8DEw7odPaic0"

    extract_images_from_pdf(pdf_path, image_dir)
    process_images_for_text(image_dir, output_file, subscription_key, endpoint)

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    with open(output_file, "r", encoding="utf-8") as file:
        text = file.read()
        text_chunks = [text[i:i + 500] for i in range(0, len(text), 500)]

    create_faiss_index(text_chunks, model, tokenizer)
    index = faiss.read_index("faiss_index.bin")

    while True:
        query = input("Ask a question: ")
        retrieved_texts = search_in_faiss(query, text_chunks, index, tokenizer, model)
        context = "\n".join(retrieved_texts)
        answer = generate_gemini_response(query, context, api_key)
        print("\n📌 **Structured Answer:**")
        # print(context)
        print(answer)



if __name__ == "__main__":
    main()

