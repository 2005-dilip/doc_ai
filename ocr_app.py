import streamlit as st
import os
import faiss
import time
from transformers import AutoTokenizer, AutoModel
from ocr import extract_images_from_pdf, process_images_for_text, create_faiss_index, search_in_faiss, \
    generate_gemini_response

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def main1():
    st.title("📖 PDF Q&A Bot")

    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
        st.session_state.text_chunks = None
        st.session_state.chat_history = []  # Store chat history

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None and st.session_state.faiss_index is None:
        start_time = time.time()
        pdf_path = os.path.join("uploads", uploaded_file.name)
        image_dir = "images"
        output_file = "output.txt"

        os.makedirs("uploads", exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        print(pdf_path)

        subscription_key = "BH5xbCyGRDbOEjhOXQuKZy2u5DMv2ioUQxjwaWx0pGhDkriwLJdLJQQJ99ALACGhslBXJ3w3AAAFACOGdWuH"
        endpoint = "https://chennai.cognitiveservices.azure.com/"

        st.info("Processing PDF... Please wait.")
        extract_images_from_pdf(pdf_path, image_dir)
        process_images_for_text(image_dir, output_file, subscription_key, endpoint)



        with open(output_file, "r", encoding="utf-8") as file:
            text = file.read()
            st.session_state.text_chunks = [text[i:i + 500] for i in range(0, len(text), 500)]

        create_faiss_index(st.session_state.text_chunks, model, tokenizer)
        st.session_state.faiss_index = faiss.read_index("faiss_index.bin")

        processing_time = time.time() - start_time
        st.success(f"PDF processing complete! You can now ask questions. (Time taken: {processing_time:.2f} seconds)")

    query = st.text_input("Ask a question:")
    if query and st.session_state.faiss_index is not None:
        query_start_time = time.time()
        retrieved_texts = search_in_faiss(query, st.session_state.text_chunks, st.session_state.faiss_index, tokenizer,
                                          model)
        context = "\n".join(retrieved_texts)
        answer = generate_gemini_response(query, context, api_key="AIzaSyBp4rx0mN65nLJxAJIsphq8DEw7odPaic0")
        query_time = time.time() - query_start_time

        print(answer)

        # Append query and answer to chat history
        st.session_state.chat_history.append((query, answer, query_time))

        # Display chat history
        st.subheader("📜 Chat History:")
        for q, a, t in st.session_state.chat_history:
            st.markdown(f"**📝 Question:** {q}")
            st.write(f"📌 **Answer:** {a}")
            st.info(f"⏳ Response time: {t:.2f} seconds")
            st.markdown("---")


if __name__ == "__main__":
    main1()
