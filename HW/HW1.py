import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from openai import OpenAI, AuthenticationError
from pypdf import PdfReader

def read_pdf(file: UploadedFile) -> str:
    """Read a PDF uploaded to the Streamlit app via st.file_uploader.

    Args:
        file: The PDF file that was uploaded to the Streamlit app.

    Returns:
        One string containing text from all pages of the uploaded PDF.
    """
    #Extracting the pages from the uploaded file object
    pages = PdfReader(file).pages

    pages_text = []

    #Iterate over each page
    for page in pages:

        #Extract text from page
        text = page.extract_text()

        #Append page text to list
        pages_text.append(text)

    # Once loop is done running, combine the text of all the pages together
    return " ".join(pages_text)

# Show title and description.
st.title("MY Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    try:
        # Create an OpenAI client.
        client = OpenAI(api_key=openai_api_key)

        client.models.list()
        
    except AuthenticationError:
        st.info("🚨 Invalid OpenAI API Key")
        st.stop()

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:

        
        # Process the uploaded file and question.
        
        file_extension = uploaded_file.name.split('.')[-1]

        if file_extension == 'txt':
         
         document = uploaded_file.read().decode()

        #If uploaded file is a PDF, text needs to be extracted from each page
        elif file_extension == 'pdf':

            document = read_pdf(uploaded_file)

        else:
            
            st.error("Unsupported file type.")


        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)
