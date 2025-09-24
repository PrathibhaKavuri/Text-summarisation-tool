>Summarize pasted text or uploaded **PDF/DOCX/TXT** files into clear,concise summaries using **BART**,**T5** or **Traditional (TF-IDF + SpaCy)** methods.Helpful for quickly digesting articles,reports or documents.

```
python -m venv venv
venv\Scripts\activate
```
```
pip install -r requirements.txt
```

# Run with Streamlit:  
```
streamlit run app.py
```

>Paste text or upload files (**PDF,DOCX,TXT**)  
>Choose summarization method:**BART**,**T5** or **Traditional**  
>Select summary length:**Short**,**Medium**,**Long** or **Custom**  
>Preview and edit summaries inside the app  
>Export summaries as **PDF** or **TXT**  
