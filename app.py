import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, BartTokenizer
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
import PyPDF2
from fpdf import FPDF
from langdetect import detect
from fpdf.enums import XPos, YPos
import docx
import os

nltk.download('wordnet')
nltk.download('punkt', quiet=True)
st.set_page_config(page_title="Text Summarization Tool", layout="centered")

@st.cache_resource(show_spinner=False)
def load_models():
    device = 0 if torch.cuda.is_available() else -1
    bart_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    t5_tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M").to('cuda' if torch.cuda.is_available() else 'cpu')
    return bart_pipeline, bart_tokenizer, t5_tokenizer, t5_model

def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()
bart_pipeline, bart_tokenizer, t5_tokenizer, t5_model = load_models()

originality_factor_map = {"Short": 0.40, "Medium": 0.60, "Long": 0.80}

def truncate_text(text, max_chars=3000):
    return text[:max_chars] if len(text) > max_chars else text

def get_token_limits_with_originality(text, length_option):
    approx_tokens = max(len(text) // 4, 10)
    factor = originality_factor_map.get(length_option, 0.6)
    max_tokens = max(50, min(int(approx_tokens * factor), 512))
    min_tokens = int(max_tokens * 0.5)
    return max_tokens, min_tokens

def summarize_t5(text, max_length, min_length):
    input_ids = t5_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True).to(t5_model.device)
    summary_ids = t5_model.generate(input_ids, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_custom_all(text, num_sentences=5):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    lemmatizer = WordNetLemmatizer()
    lemmatized_sentences = []
    for sent in sentences:
        words = word_tokenize(sent)
        lemmas = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]
        lemmatized_sentences.append(' '.join(lemmas))

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(lemmatized_sentences)
    tfidf_scores = tfidf_matrix.sum(axis=1).A1

    doc = nlp(text)
    named_entities = set(ent.text.lower() for ent in doc.ents if ent.label_)

    boosted_scores = []
    for idx, sent in enumerate(sentences):
        score = tfidf_scores[idx]
        if any(ent in sent.lower() for ent in named_entities):
            score += 0.2
        boosted_scores.append((score, sent))

    top_sentences = sorted(boosted_scores, reverse=True)[:num_sentences]
    ordered_summary = sorted([s for _, s in top_sentences], key=sentences.index)

    return ' '.join(ordered_summary)

def read_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    full_text = [page.extract_text() for page in pdf.pages if page.extract_text()]
    return ' '.join(full_text)

def read_docx(file):
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def read_txt(file):
    return file.read().decode('utf-8')  

def generate_pdf(summary, style="Paragraph"):
    pdf = FPDF()
    pdf.add_page()

    font_path = "fonts/DejaVuSans.ttf"
    if not os.path.exists(font_path):
        raise FileNotFoundError("Font file 'DejaVuSans.ttf' not found in 'fonts/' directory.")

    pdf.add_font('DejaVu', '', font_path, uni=True)
    pdf.set_font('DejaVu', '', 14)

    pdf.set_text_color(30, 30, 60)
    pdf.cell(0, 15, "Text Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(10)

    pdf.set_text_color(0, 0, 0)
    line_height = 8
    left_margin = 15
    bullet_indent = 8
    pdf.set_left_margin(left_margin)
    pdf.set_right_margin(left_margin)

    if style == "Paragraph":
        pdf.multi_cell(0, line_height, summary, align='L')
    elif style == "Bulleted List":
        sentences = sent_tokenize(summary)
        bullet = "•"
        for sent in sentences:
            pdf.cell(bullet_indent, line_height, bullet)
            pdf.multi_cell(0, line_height, sent, align='L')
            pdf.ln(2)

    return bytes(pdf.output(dest='S'))

def reset_summary():
    st.session_state["summary"] = ""
    st.session_state["show_summary"] = False

if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "show_summary" not in st.session_state:
    st.session_state["show_summary"] = False
if "last_method" not in st.session_state:
    st.session_state["last_method"] = None
if "last_summary_length" not in st.session_state:
    st.session_state["last_summary_length"] = None
if "last_pdf_format" not in st.session_state:
    st.session_state["last_pdf_format"] = None
if "last_input_type" not in st.session_state:
    st.session_state["last_input_type"] = None
if "last_text_input" not in st.session_state:
    st.session_state["last_text_input"] = ""


with st.sidebar:
    st.markdown("### Options")
    method = st.selectbox("Summarization Method", ["BART", "T5", "Traditional"])
    summary_length = st.selectbox("Summary Length", ["Short", "Medium", "Long", "Custom"])
    custom_sentences = 5
    if summary_length == "Custom":
        custom_sentences = st.number_input("Number of sentences for Custom Summary", min_value=1, max_value=50, value=5)
    pdf_format = st.selectbox("PDF Format Style", ["Paragraph", "Bulleted List"])

    st.markdown("#### Not Sure Which Summarization Method to Use?")

    with st.expander("Quick & Easy Guide to Summarization"):
        st.markdown("""
     **Abstractive Summarization**  
    This method **writes a brand-new summary** in its own words — kind of like how you’d explain something to a friend.

    - **BART (Facebook):** Great for long texts like reports or articles. It understands the full story and makes it sound smooth.  
    - **T5 (Google):** Fast and reliable, works well on most types of writing.

     **Traditional Summarization (Extractive Method)**  
    This method **picks out the most important sentences** right from your original text. It uses smart tools like SpaCy and TF-IDF to find key points without changing the words.

    - **Traditional:** Quick and simple — perfect if you want to see the main ideas exactly as they appear in your text. Great for notes, emails, or shorter documents.

    ---
     **Quick Tips:**  
    - Pick **BART** or **T5** if you want a fresh, easy-to-read summary.  
    - Pick **Traditional** if you want a fast summary that shows the important parts from your original text.

    Give them a try and see which one feels right for you!
    """)

st.markdown("<h1 style='text-align: center;'>Text Summarizer</h1>", unsafe_allow_html=True)
st.markdown("Summarize pasted text or uploaded PDFs using BART, T5, or NLTK methods.")
input_type = st.radio("Input Method", ["Paste text", "Upload File"])
st.markdown('<small>Note:Summary length is appr. and depends on the input text and method.</small>', unsafe_allow_html=True)

if (
    st.session_state["last_method"] != method
    or st.session_state["last_summary_length"] != summary_length
    or st.session_state["last_pdf_format"] != pdf_format
    or st.session_state["last_input_type"] != input_type
):
    reset_summary()
    st.session_state["last_method"] = method
    st.session_state["last_summary_length"] = summary_length
    st.session_state["last_pdf_format"] = pdf_format
    st.session_state["last_input_type"] = input_type

text_input, uploaded_file = "", None
if input_type == "Paste text":
    text_input = st.text_area("Enter text to summarize", height=250)
    if st.session_state["last_text_input"] != text_input:
        reset_summary()
        st.session_state["last_text_input"] = text_input
else:
    uploaded_file = st.file_uploader("Upload file", type=["pdf", "docx", "txt"])
    if uploaded_file:
        try:
            if uploaded_file.type == "application/pdf":
                text_input = read_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text_input = read_docx(uploaded_file)
            elif uploaded_file.type == "text/plain":
                text_input = read_txt(uploaded_file)
            else:
                st.error("Unsupported file type.")
            if st.session_state["last_text_input"] != text_input:
                reset_summary()
                st.session_state["last_text_input"] = text_input
        except Exception as e:
            st.error(f"Error reading file: {e}")    
            text_input = ""
        

if st.button("Generate Summary") and text_input.strip() and len(text_input.strip().split()) >= 10:
    with st.spinner("Generating summary..."):
        try:
            if detect(text_input) != 'en':
                st.warning("Only English input is supported currently.")
            truncated_text = truncate_text(text_input)
            max_tokens, min_tokens = get_token_limits_with_originality(text_input, summary_length)

            if method == "BART":
                summary = bart_pipeline(truncated_text, max_length=max_tokens, min_length=min_tokens, do_sample=False)[0]['summary_text']
            elif method == "T5":
                summary = summarize_t5(truncated_text, max_length=max_tokens, min_length=min_tokens)
            else:
                num_sent = custom_sentences if summary_length == "Custom" else {"Short": 5, "Medium": 7, "Long": 10}.get(summary_length, 5)
                summary = summarize_custom_all(text_input, num_sent)

            st.session_state["summary"] = summary
            st.session_state["show_summary"] = True
            st.success("Summary generated!")

        except Exception as e:
            st.error(f"Summarization failed: {e}")

if st.session_state["show_summary"]:
    st.markdown("### Edit the summary text below:")
    edited_summary = st.text_area("Summary (editable)", value=st.session_state["summary"], height=200)
    st.session_state["summary"] = edited_summary

    st.markdown("### Preview of formatted summary:")
    if pdf_format == "Paragraph":
        st.markdown(edited_summary.replace('\n', '  \n'))
    else:
        bullets_md = '\n'.join([f"- {s}" for s in sent_tokenize(edited_summary)])
        st.markdown(bullets_md)

    pdf_bytes = generate_pdf(edited_summary, style=pdf_format)
    pdf_data = generate_pdf(edited_summary, style=pdf_format)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download Summary as PDF",
            data=pdf_data,
            file_name="summary.pdf",
            mime="application/pdf"
        )
    with col2:
        txt_data = edited_summary.encode('utf-8')
        st.download_button(
            label="Download Summary as TXT",
            data=txt_data,
            file_name="summary.txt",
            mime="text/plain"
        )
