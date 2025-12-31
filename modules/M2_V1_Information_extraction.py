

import os
openai.api_key = os.getenv("OPENAI_API_KEY")



from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Say hello in one sentence."}
    ]
)

print(response.choices[0].message.content)


from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import re
import json
from typing import List, Dict
import os






def extract_legal_sections(text):
    print("inside legal section")
    pattern = r"(Section\s+\d+[A-Z]*\s*(?:of)?\s*[A-Za-z ]+)"
    return list(set(re.findall(pattern, text, re.IGNORECASE)))


def extract_legal_references(text):
    """
    Extracts legal references like Sections, Clauses, Articles, Rules, and Act/Law names from text.
    Returns a list of matches.
    """
    pattern = r"""
    # Match Section, Clause, Article, Rule references
    (?:\b(Section|Clause|Article|Rule)\s*       # Keyword
    \d+[A-Z]?                                   # Number with optional letter
    (?:\(\d+\))?                                # Optional subsection like (1)
    \s*(?:of)?\s*                               # Optional 'of'
    [A-Za-z ]+)                                 # Name of the Act/Law

    |                                           # OR

    # Match general Act/Law names not preceded by Section/Clause etc.
    \b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)         # Capitalized multi-word names
    """

    matches = re.findall(pattern, text, re.VERBOSE)

    # Flatten the result and remove empty strings
    result = [m[0] if m[0] else m[1] for m in matches]

    return result


def extract_dates(text):
    date_patterns = [
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        r"\b\d{4}\b"
    ]
    dates = []
    for p in date_patterns:
        dates.extend(re.findall(p, text))
    return list(set(dates))




from openai import OpenAI
import json

client = OpenAI()

def extract_legal_information_llm(text):
    prompt = f"""
You are a legal document analysis assistant.

Extract the following information from the text below.incase of native laguage , please convert it to english and read and provide the output in english
Return STRICT JSON only. while printing the results please do not use the Actual names, provide as Respondent/Petioner alone.

Fields:
- court_name
- case_number
- sections_invoked
- important_dates
- reliefs_sought
- events
- persons
- legal_references
- legal_sections
- legal_acts
- legal_laws


Text:
\"\"\"{text}\"\"\"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract structured legal information."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content





from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import json
import os
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
FAISS_DIR = f"/content/drive/MyDrive/legal_embeddings_db"
vs = FAISS.load_local(FAISS_DIR, emb, allow_dangerous_deserialization=True)

retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 200,
        "filter": {"case_id": "GWOP_3261"}
    }
)
queries = [
    "case number",
    "date of marriage",
    "date of separation",
    "dowry or streedhan details",
    "child custody information",
    "court name and judge",
    "Transacation done by the Petioner/Respondent with date",
    "All Money related transactions",
    "Allegations provided by the petitioner/Respondent"

]

all_docs = []
for q in queries:
    all_docs.extend(retriever.invoke(q))
print(len(all_docs))
#print(docs[0].page_content if docs else "NO RESULTS")



retrieved_text = "\n\n".join([doc.page_content for doc in all_docs])
legal_info = extract_legal_information_llm(retrieved_text)
case_id = "GWOP_3261" #should make the changes to pull the case id dynamically

output = {
    "case_id": case_id,
    "retrieved_text": retrieved_text,
    "extracted_info": legal_info
}
#ai_output_path = f"/content/drive/MyDrive/legal_ai/structured_outputs/"
#os.makedirs(ai_output_path, exist_ok=True)
output_path = f"/content/drive/MyDrive/legal_ai/structured_outputs/{case_id}.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("Saved to:", output_path)



client = OpenAI()

def legal_rag_query(query, case_id, base_path="/content/drive/MyDrive/legal_ai/structured_outputs"):
    case_file = f"{base_path}/{case_id}.json"

    if not os.path.exists(case_file):
        return {"error": f"Case {case_id} not found"}

    with open(case_file, "r") as f:
        case_data = json.load(f)

    context = case_data.get("retrieved_text", "")

    prompt = f"""
You are a legal assistant.

Use ONLY the information below to answer the question.
If the answer is not present, say "Not found in record".

CASE DATA:
{context}

QUESTION:
{query}

Answer concisely and factually.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content


answer = legal_rag_query(query_input,case_id_input)
   

print(answer)



# Inspect 3 random vectors
docs = vs.similarity_search("test", k=20)

for d in docs:
    print("METADATA:", d.metadata)
    print("TEXT:", d.page_content[:200])
    print("-" * 40)


retriever = vs.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {
            "case_id": "DVC_74"
        }
    }
)


# Inspect 3 random vectors
docs = vs.similarity_search("test", k=20)

for d in docs:
    print("METADATA:", d.metadata)
    print("TEXT:", d.page_content[:200])
    print("-" * 40)

from openai import OpenAI
client = OpenAI()

def extract_events(text):
    prompt = f"""
You are a legal analyst.

Extract all important life or case events from the text.
Return ONLY valid JSON array.

Each item must contain:
- event
- date (or "Unknown")
- source (if mentioned)

Text:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content


from datetime import datetime

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return None

def sort_events(events):
    return sorted(
        events,
        key=lambda x: parse_date(x["date"]) or datetime.max
    )


def detect_inconsistencies(events):
    issues = []

    for i in range(len(events)-1):
        if events[i]["date"] == "Unknown":
            issues.append(f"Missing date for event: {events[i]['event']}")

        if events[i]["date"] > events[i+1]["date"]:
            issues.append(
                f"Chronology issue: '{events[i]['event']}' occurs after '{events[i+1]['event']}'"
            )

    return issues


from typing import List, Dict

Event = {
    "event": str,
    "date": str,      # YYYY or DD-MM-YYYY or "Unknown"
    "source": str     # document name or page
}


import json
import re

def safe_json_load(text):
    # Extract JSON block if wrapped in ```json
    match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text)

    if not match:
        raise ValueError("No JSON found in LLM output")

    json_str = match.group(0)

    return json.loads(json_str)



events = extract_events(retrieved_text)
raw_output = extract_events(retrieved_text)
#print("RAW OUTPUT ↓↓↓")
#print(raw_output)

# Convert string → list[dict]
events = safe_json_load(raw_output)

events = sort_events(events)
issues = detect_inconsistencies(events)

print("Timeline:")
for e in events:
    print(e)

print("\nIssues:")
for i in issues:
    print("-", i)
