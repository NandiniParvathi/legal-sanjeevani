# ===============================
# 1. Install & Setup
# ===============================
import os
from google.colab import drive
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from pdf2image import convert_from_path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pytesseract

drive.mount('/content/drive')

BASE_DIR = "/content/drive/MyDrive"
DATA_DIR = f"{BASE_DIR}/Court_docs"
FAISS_DIR = f"/content/drive/MyDrive/legal_embeddings_db"

os.makedirs(FAISS_DIR, exist_ok=True)
print("Data dir:", DATA_DIR)
print("FAISS_DIR:", FAISS_DIR)

# ===============================
# Defining the OCR for Ingesting Legal Documents
# ===============================


def save_ocr_text(text, output_path):

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def load_pdf_text_ocr(file_path):
    print("Loading PDF text with OCR...")
    pages = convert_from_path(file_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page) + "\n"
    return text.strip()

def load_pdf_text_smart(file_path,case_id):
    print("Loading PDF text with OCR (scanned PDF)...")
    text = load_pdf_text_ocr(file_path)
    ocr_text_path = f"/content/drive/MyDrive/ocr_texts/{case_id}.txt"
    save_ocr_text(text, ocr_text_path)
    print("OCR text saved to:", ocr_text_path)
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # good for legal docs
        chunk_overlap=200
    )
    return splitter.split_text(text)

def ingest_pdf(file_path, case_id, doc_type,vectorstore,emb):

    print(f"Ingesting {file_path}...")

    text = load_pdf_text_smart(file_path,case_id)
    print("Extracted text length:", len(text))

    if not text or len(text) < 100:
        raise ValueError(f"OCR failed or empty text for {file_path}")

    chunks = chunk_text(text)
    print(f"Total chunks created: {len(chunks)}")

    if len(chunks) == 0:
        raise ValueError("No chunks generated")

    metadatas = []
    ids = []
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "case_id": case_id,
                    "doc_type": doc_type,
                    "source": os.path.basename(file_path),
                    "chunk_id": i
                }
            )
        )
    if vectorstore is None:
        print("🆕 Creating FAISS index")
        vectorstore = FAISS.from_documents(docs, emb)
    else:
        vectorstore.add_documents(docs)


    print(f"✅ Embedded & stored {len(docs)} chunks from {doc_type}")
    return vectorstore

# ===============================
# Ingestion Uploaded/Reading the court documents from Drive
# ===============================
#
#global declaration of FAISS embeddings
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vs=None
#if os.path.exists(FAISS_DIR):
#    vs = FAISS.load_local(FAISS_DIR, emb,allow_dangerous_deserialization=True)
#else:
#    vs=None
ai_text_path = f"/content/drive/MyDrive/ocr_texts/"
os.makedirs(ai_text_path, exist_ok=True)
print("Ingesting Legal Documents")
print("Ingesting GWOP Petition...")
pdf_path_gwop1 = f"/content/drive/MyDrive/Court_docs/GWOP-Highcourt/gwop 3261 petition.pdf"
vs=ingest_pdf(
    file_path=pdf_path_gwop1,
    case_id="GWOP_3261",
    doc_type="petition",
    vectorstore=vs,
    emb=emb
)
vs.save_local(FAISS_DIR)
print("✅ FAISS index saved")
print("Files in FAISS_DIR:", os.listdir(FAISS_DIR))

print("Ingesting GWOP Counter...")
pdf_path_gwop2 = f"/content/drive/MyDrive/Court_docs/GWOP-Highcourt/gwop counter.pdf"
vs=ingest_pdf(
    file_path=pdf_path_gwop2,
    case_id="GWOP_3261",
    doc_type="counter",
    vectorstore=vs,
    emb=emb
)
vs.save_local(FAISS_DIR)
print("✅ FAISS index saved")
print("Files in FAISS_DIR:", os.listdir(FAISS_DIR))

print("Ingesting HMOP Petition...")
pdf_path_gwop1 = f"/content/drive/MyDrive/Court_docs/HMOP- Alandur/HMOP1885-2021.pdf"
vs=ingest_pdf(
    file_path=pdf_path_gwop1,
    case_id="HMOP_1885",
    doc_type="petition",
    vectorstore=vs,
    emb=emb
)
vs.save_local(FAISS_DIR)
print("✅ FAISS index saved")
print("Files in FAISS_DIR:", os.listdir(FAISS_DIR))

print("Ingesting HMOP Counter...")
pdf_path_gwop1 = f"/content/drive/MyDrive/Court_docs/HMOP- Alandur/HMOP- counter.pdf"
vs=ingest_pdf(
    file_path=pdf_path_gwop1,
    case_id="HMOP_1885",
    doc_type="counter",
    vectorstore=vs,
    emb=emb
)
vs.save_local(FAISS_DIR)
print("✅ FAISS index saved")
print("Files in FAISS_DIR:", os.listdir(FAISS_DIR))

print("Ingesting DV Petition...")
pdf_path_gwop1 = f"/content/drive/MyDrive/Court_docs/DVC- Saidapet/dvc petition74-2023.pdf"
vs=ingest_pdf(
    file_path=pdf_path_gwop1,
    case_id="DVC_74",
    doc_type="petition",
    vectorstore=vs,
    emb=emb
)
vs.save_local(FAISS_DIR)
print("✅ FAISS index saved")
print("Files in FAISS_DIR:", os.listdir(FAISS_DIR))

print("Ingesting DVC Counter...")
pdf_path_gwop1 = f"/content/drive/MyDrive/Court_docs/DVC- Saidapet/counter dvc.pdf"
vs=ingest_pdf(
    file_path=pdf_path_gwop1,
    case_id="DVC_74",
    doc_type="counter",
    vectorstore=vs,
    emb=emb
)
vs.save_local(FAISS_DIR)
print("✅ FAISS index saved")
print("Files in FAISS_DIR:", os.listdir(FAISS_DIR))




