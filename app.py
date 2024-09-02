import streamlit as st
from transformers import pipeline
import PyPDF2
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Function to create a word cloud
def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(plt)

# Main function
def main():
    # Set the page title and icon
    st.set_page_config(page_title="Advanced Text Summarizer", page_icon="📝", layout="wide")

    # Display a header image
    st.image("nust background.jpg", use_column_width=True)  # Replace with your image path

    # Title with an emoji
    st.title("📝 Advanced Text Summarizer")

    # Sidebar with a background color
    st.sidebar.markdown("## Navigation")
    st.sidebar.markdown("### Select Input Method:")
    
    # Option to upload a PDF or enter text
    option = st.sidebar.radio("Choose input method", ["Text Input", "Upload PDF"])
    
    st.sidebar.image("NUST_Vector.svg", use_column_width=True)  # Add an image in the sidebar

    st.markdown("---")  # Section divider

    # Model selection
    st.sidebar.markdown("### Model Parameters:")
    min_len = st.sidebar.slider('Minimum Summary Length', 10, 100, 20)
    max_len = st.sidebar.slider('Maximum Summary Length', 50, 500, 250)

    summarizer = pipeline("summarization", 
                          model="t5-small", 
                          min_length=min_len, 
                          max_length=max_len)

    # Main content layout
    input_text = ""
    col1, col2 = st.columns(2)

    with col1:
        if option == "Text Input":
            st.subheader("Enter Text to Summarize")
            input_text = st.text_area('Enter your text here:')
            if input_text:
                st.info(f'Word Count: {len(input_text.split(" "))}')

        elif option == "Upload PDF":
            st.subheader("Upload a PDF File")
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            if uploaded_file is not None:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                num_pages = len(pdf_reader.pages)
                for page in range(num_pages):
                    input_text += pdf_reader.pages[page].extract_text()
            if input_text:
                st.info(f'Word Count: {len(input_text.split(" "))}')

    with col2:
        if st.button('Summarize'):
            if input_text:
                output = summarizer(input_text)
                summary = output[0]['summary_text']
                st.success(f'Summary Word Count: {len(summary.split(" "))}')
                st.subheader("Summary:")

                bullet_points = summary.split('. ')
                for point in bullet_points:
                    st.markdown(f"• **{point}**")

                # Display a word cloud of the summary
                st.subheader("Word Cloud of the Summary:")
                create_wordcloud(summary)

                # Option to save the summary
                st.download_button("Download Summary", summary, file_name="summary.txt")

    st.markdown("---")  # Section divider

    # Translation section after summarization
    if input_text and st.button('Translate to Urdu'):
        summarizer_output = summarizer(input_text)
        summary = summarizer_output[0]['summary_text']
        
        st.subheader("Translate Summary to Urdu:")
        translater = pipeline('translation', model='Helsinki-NLP/opus-mt-en-ur')
        translated_text = translater(summary)[0]['translation_text']

        bullet_points = translated_text.split('-')
        for point in bullet_points:
            st.markdown(f"• **{point}**")

        st.download_button('Download Translated Summary', translated_text, file_name='summary_urdu.txt')

if __name__ == '__main__':
    main()
