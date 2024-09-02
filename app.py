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

    # Initialize session state
    if 'summary' not in st.session_state:
        st.session_state.summary = ""

    if 'translated_summary' not in st.session_state:
        st.session_state.translated_summary = ""

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
                st.session_state.summary = output[0]['summary_text']
                st.success(f'Summary Word Count: {len(st.session_state.summary.split(" "))}')
                st.subheader("Summary:")

                # Split summary into bullet points
                bullet_points = st.session_state.summary.split('. ')
                for point in bullet_points:
                    st.markdown(f"• **{point}**")

                # Display a word cloud of the summary
                st.subheader("Word Cloud of the Summary:")
                create_wordcloud(st.session_state.summary)

                # Option to save the summary
                st.download_button("Download Summary", st.session_state.summary, file_name="summary.txt", key="summary_download")

    st.markdown("---")  # Section divider

    # Translation section
    if st.session_state.summary and st.button('Translate to Urdu'):
        translater = pipeline('translation_en_to_ur', model='Helsinki-NLP/opus-mt-en-ur')
        st.session_state.translated_summary = translater(st.session_state.summary)[0]['translation_text']

        st.subheader("Translate Summary to Urdu:")
        bullet_points = st.session_state.translated_summary.split('-')  # Split on Urdu sentence delimiter
        for point in bullet_points:
            if point.strip():
                st.markdown(f"• **{point.strip()}**")

        st.download_button('Download Translated Summary', st.session_state.translated_summary, file_name='summary_urdu.txt', key='translated_summary_download')

    # Display the summaries if they exist
    if st.session_state.summary:
        st.subheader("Summary:")
        bullet_points = st.session_state.summary.split('. ')
        for point in bullet_points:
            st.markdown(f"• **{point}**")

        st.download_button("Download Summary", st.session_state.summary, file_name="summary.txt", key="summary_download_repeated")

   

if __name__ == '__main__':
    main()
