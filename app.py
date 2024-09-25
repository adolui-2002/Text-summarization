import streamlit as st
from transformers import pipeline
from newspaper import Article
import easyocr
from PIL import Image
import numpy as np

st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #032c40;  
    }
    footer {
        visibility: hidden;  # Hide the default Streamlit footer
    }
    .custom-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #001f3f;  # Matte navy blue
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #ffffff;  # White text for contrast
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Article Summarizer")

st.markdown(
    """
    <div class="custom-footer">Aditya Vishal Tiwari   |   Padmendra Singh Yadav  |   Pranav Kumar  |  
        Arunima Dolui  |   Nitish Kumar Ray  |   Projyoti Barik
    </div>
    """,
    unsafe_allow_html=True,
)

pipe = pipeline("summarization", model="google/pegasus-xsum")

reader = easyocr.Reader(['en']) 

summary_type = st.radio("Summarize from:", ["Text Input", "URL", "Image (OCR)"])

if summary_type == "Text Input":
    input_text = st.text_area("Enter text to summarize:", height=150)
    if st.button("Summarize"):
        try:
            query = input_text + "\nTL;DR:\n"
            pipe_out = pipe(query, max_length=100, clean_up_tokenization_spaces=True)
            summary = pipe_out[0]["summary_text"]
            st.write("Summary:")
            st.write(summary)
        except Exception as e:
            st.write("Error summarizing the text. Please try again.")

elif summary_type == "URL":
    url = st.text_input("Enter URL to summarize:")
    if st.button("Fetch and Summarize"):
        if url and url.startswith(("http://", "https://")):
            try:
                article = Article(url)
                article.download()
                article.parse()
                input_text = article.text
                query = input_text + "\nTL;DR:\n"
                pipe_out = pipe(query, max_length=100, clean_up_tokenization_spaces=True)
                summary = pipe_out[0]["summary_text"]
                st.write("Summary:")
                st.write(summary)
            except Exception as e:
                st.write("Error fetching or summarizing the article. It might be protected against scraping or is not valid. Please try another URL.")
        else:
            st.write("Please enter a valid URL (starting with http:// or https://).")

elif summary_type == "Image (OCR)":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_np = np.array(image)
        
        with st.spinner('Performing OCR...'):
            results = reader.readtext(image_np, detail=0)  
            extracted_text = ' '.join(results)
        
        if extracted_text:
            st.write("### Extracted Text:")
            st.write(extracted_text)
            
            if st.button("Summarize OCR Text"):
                try:
                    query = extracted_text + "\nTL;DR:\n"
                    pipe_out = pipe(query, max_length=100, clean_up_tokenization_spaces=True)
                    summary = pipe_out[0]["summary_text"]
                    st.write("Summary:")
                    st.write(summary)
                except Exception as e:
                    st.write("Error summarizing the OCR text. Please try again.")
