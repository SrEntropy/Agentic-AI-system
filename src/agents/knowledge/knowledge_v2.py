import os
import json
import sqlite3
import requests
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

# Third-party imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

def load_env():
    from dotenv import load_dotenv
    load_dotenv()

# --- Report Scraper ---
class ReportScraper:
    """Scrapes public vulnerability reports from various sources."""
    def scrape_swc(self) -> List[Dict]:
        url = "https://swcregistry.io/"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        findings = []
        for swc in soup.select("div.swc-entry"):
            swc_id = swc.select_one("h3").text.strip()
            title = swc.select_one("h4").text.strip()
            desc = swc.select_one("p").text.strip()
            findings.append({
                "identifier": "public",
                "tool": "SWC Registry",
                "severity": "Varies",
                "title": title,
                "description": desc,
                "location": "N/A",
                "swc_id": swc_id,
                "source": "SWC"
            })
        return findings

    def scrape_immunefi(self) -> List[Dict]:
        url = "https://immunefi.com/vulnerability/"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        findings = []
        for report in soup.select("div.report"):
            title = report.select_one("h2").text.strip()
            desc = report.select_one("p").text.strip()
            findings.append({
                "identifier": "public",
                "tool": "Immunefi",
                "severity": "High",
                "title": title,
                "description": desc,
                "location": "N/A",
                "swc_id": "Unknown",
                "source": "Immunefi"
            })
        return findings

    def scrape_all(self) -> List[Dict]:
        return self.scrape_swc() + self.scrape_immunefi()

# --- Findings Parser ---
class FindingsParser:
    """Parses AnalyzerAgent output into normalized records."""
    def parse(self, findings_path: str) -> List[Dict]:
        findings = []
        with open(findings_path, "r") as f:
            data = json.load(f)
        identifier = os.path.basename(findings_path).replace(".json", "")
        for item in data:
            findings.append({
                "identifier": identifier,
                "tool": item.get("tool", "Unknown"),
                "severity": item.get("severity", "Unknown"),
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "location": item.get("location", ""),
                "swc_id": item.get("swc_id", "N/A"),
                "source": "Analyzer"
            })
        return findings

# --- Storage Backend ---
class FindingsStorage:
    """Stores findings to SQLite and ChromaDB."""
    def __init__(self, base_dir: str):
        self.db_path = os.path.join(base_dir, "vulnerabilities.db")
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()
        self.vectorstore = Chroma(
            collection_name="knowledge",
            embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2"),
            persist_directory=os.path.join(base_dir, "chromadb")
        )

    def _init_db(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vulnerabilities (
                id INTEGER PRIMARY KEY,
                identifier TEXT,
                tool TEXT,
                severity TEXT,
                title TEXT,
                description TEXT,
                location TEXT,
                swc_id TEXT,
                source TEXT
            )
            """
        )
        self.conn.commit()

    def store(self, findings: List[Dict]):
        # SQLite
        for f in findings:
            self.conn.execute(
                "INSERT INTO vulnerabilities (identifier, tool, severity, title, description, location, swc_id, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f['identifier'], f['tool'], f['severity'], f['title'], f['description'], f['location'], f['swc_id'], f['source'])
            )
        self.conn.commit()
        # ChromaDB
        docs = [f"{f['title']}: {f['description']}" for f in findings]
        metas = [{"identifier": f['identifier'], "source": f['source']} for f in findings]
        if docs:
            self.vectorstore.add_texts(texts=docs, metadatas=metas)

# --- Trainer ---
class VulnerabilityTrainer:
    """Prepares data and fine-tunes an LLM for vulnerability detection."""
    def __init__(self, model_name: str, base_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.base_dir = base_dir

    def prepare_dataset(self, findings: List[Dict], public: List[Dict]) -> List[Dict]:
        data = []
        for f in findings + public:
            data.append({"text": f["description"], "label": 1})
        # negative examples
        data += [{"text": "function safe() external {}", "label": 0}]
        return data

    def fine_tune(self, dataset: List[Dict]):
        df = pd.DataFrame(dataset)
        train_df, eval_df = train_test_split(df, test_size=0.2)
        # tokenization
        def tokenize(exs): return self.tokenizer(exs['text'], padding=True, truncation=True, max_length=128)
        train_enc = tokenize(train_df.to_dict('records'))
        eval_enc = tokenize(eval_df.to_dict('records'))
        # dataset
        # ... same as before
        # trainer
        training_args = TrainingArguments(
            output_dir=os.path.join(self.base_dir, "model"),
            num_train_epochs=3,
            per_device_train_batch_size=8,
            evaluation_strategy="epoch"
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_enc,
            eval_dataset=eval_enc,
            data_collator=DataCollatorWithPadding(self.tokenizer)
        )
        trainer.train()

# --- Predictor ---
class VulnerabilityPredictor:
    """Predicts vulnerability probability for code snippets."""
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def predict(self, snippet: str) -> float:
        inputs = self.tokenizer(snippet, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[0][1].item()

# --- KnowledgeAgent Orchestrator ---
class KnowledgeAgent:
    def __init__(self, data_dir: str = "~/agentic-ai/data"):
        load_env()
        self.base = os.path.expanduser(data_dir)
        self.parser = FindingsParser()
        self.scraper = ReportScraper()
        self.storage = FindingsStorage(os.path.join(self.base, "knowledge"))
        self.trainer = VulnerabilityTrainer("distilbert-base-uncased", os.path.join(self.base, "knowledge"))
        self.predictor = VulnerabilityPredictor(
            self.trainer.tokenizer, self.trainer.model, self.trainer.device
        )

    def run_task(self, task: Dict):
        findings = []
        if task.get("findings_path"):
            parsed = self.parser.parse(task["findings_path"])
            findings.extend(parsed)
            self.storage.store(parsed)
        if task.get("train"):
            public = self.scraper.scrape_all()
            data = self.trainer.prepare_dataset(findings, public)
            self.trainer.fine_tune(data)
        if task.get("contract_path"):
            code = open(task["contract_path"]).read()
            prob = self.predictor.predict(code)
            return {"vulnerability_prob": prob}
        return None

if __name__ == "__main__":
    agent = KnowledgeAgent()
    task = {"findings_path": "~/agentic-ai/data/findings/example.json", "train": True, "contract_path": "~/agentic-ai/data/contracts/example.sol"}
    print(agent.run_task(task))
