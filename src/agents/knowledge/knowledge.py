import os
import json
import sqlite3
import requests
from typing import Dict, List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split

load_dotenv()

class KnowledgeAgent:
    def __init__(self, data_dir: str = "~/agentic-ai/data"):
        """Initialize KnowledgeAgent with data directories, embeddings, and LLM."""
        self.data_dir = os.path.expanduser(data_dir)
        self.findings_dir = os.path.join(self.data_dir, "findings")
        self.knowledge_dir = os.path.join(self.data_dir, "knowledge")
        self.db_path = os.path.join(self.knowledge_dir, "vulnerabilities.db")
        os.makedirs(self.knowledge_dir, exist_ok=True)
        os.makedirs(self.findings_dir, exist_ok=True)

        # Initialize ChromaDB
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")
        self.vectorstore = Chroma(
            collection_name="knowledge",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(self.knowledge_dir, "chromadb")
        )

        # Initialize SQLite
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vulnerabilities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identifier TEXT,
                tool TEXT,
                severity TEXT,
                title TEXT,
                description TEXT,
                location TEXT,
                swc_id TEXT,
                source TEXT
            )
        """)
        self.conn.commit()

        # Initialize LLM
        self.model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2  # Vulnerable vs. non-vulnerable
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def parse_findings(self, findings_path: str) -> List[Dict]:
        """Parse AnalyzerAgent findings from JSON."""
        print(f"[*] Parsing findings: {findings_path}")
        findings = []
        try:
            with open(findings_path, "r") as f:
                data = json.load(f)
            for finding in data:
                normalized = {
                    "identifier": os.path.basename(findings_path).replace(".json", ""),
                    "tool": finding.get("tool", "Unknown"),
                    "severity": finding.get("severity", "Unknown"),
                    "title": finding.get("title", "Unknown issue"),
                    "description": finding.get("description", ""),
                    "location": finding.get("location", "N/A"),
                    "swc_id": finding.get("swc_id", "N/A"),
                    "source": "AnalyzerAgent"
                }
                findings.append(normalized)
            print(f"[+] Parsed {len(findings)} findings")
        except Exception as e:
            print(f"[-] Findings parsing failed: {e}")
        return findings

    def scrape_public_reports(self) -> List[Dict]:
        """Scrape public vulnerability reports from SWC Registry and Immunefi."""
        print("[*] Scraping public vulnerability reports")
        findings = []
        try:
            # SWC Registry
            response = requests.get("https://swcregistry.io/", timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for swc in soup.select("div.swc-entry"):
                swc_id = swc.select_one("h3").text.strip()  # e.g., SWC-107
                title = swc.select_one("h4").text.strip()   # e.g., Reentrancy
                description = swc.select_one("p").text.strip()
                findings.append({
                    "identifier": "public",
                    "tool": "SWC Registry",
                    "severity": "Varies",
                    "title": title,
                    "description": description,
                    "location": "N/A",
                    "swc_id": swc_id,
                    "source": "SWC Registry"
                })

            # Immunefi Disclosures (simplified example)
            response = requests.get("https://immunefi.com/vulnerability/", timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for report in soup.select("div.report"):
                title = report.select_one("h2").text.strip()
                description = report.select_one("p").text.strip()
                swc_id = "Unknown"  # Requires manual mapping or regex
                findings.append({
                    "identifier": "public",
                    "tool": "Immunefi",
                    "severity": "High",
                    "title": title,
                    "description": description,
                    "location": "N/A",
                    "swc_id": swc_id,
                    "source": "Immunefi"
                })

            print(f"[+] Scraped {len(findings)} public findings")
        except Exception as e:
            print(f"[-] Public report scraping failed: {e}")
        return findings

    def store_findings(self, findings: List[Dict]):
        """Store findings in SQLite and ChromaDB."""
        print("[*] Storing findings")
        try:
            # SQLite
            for finding in findings:
                self.conn.execute("""
                    INSERT INTO vulnerabilities (identifier, tool, severity, title, description, location, swc_id, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    finding["identifier"],
                    finding["tool"],
                    finding["severity"],
                    finding["title"],
                    finding["description"],
                    finding["location"],
                    finding["swc_id"],
                    finding["source"]
                ))
            self.conn.commit()

            # ChromaDB
            documents = [
                f"{f['title']}: {f['description']} (Severity: {f['severity']}, SWC: {f['swc_id']})"
                for f in findings
            ]
            metadatas = [{"identifier": f["identifier"], "source": f["source"]} for f in findings]
            if documents:
                self.vectorstore.add_texts(texts=documents, metadatas=metadatas)
                print(f"[+] Indexed {len(documents)} findings in ChromaDB")
        except Exception as e:
            print(f"[-] Findings storage failed: {e}")

    def prepare_training_data(self) -> List[Dict]:
        """Prepare dataset for LLM fine-tuning."""
        print("[*] Preparing training data")
        dataset = []
        try:
            # Load AnalyzerAgent findings
            for findings_file in os.listdir(self.findings_dir):
                if findings_file.endswith(".json"):
                    findings = self.parse_findings(os.path.join(self.findings_dir, findings_file))
                    for finding in findings:
                        if finding["swc_id"] != "N/A":
                            dataset.append({
                                "text": f"{finding['description']} {finding['location']}",
                                "label": 1  # Vulnerable
                            })

            # Load public reports
            public_findings = self.scrape_public_reports()
            for finding in public_findings:
                dataset.append({
                    "text": f"{finding['description']}",
                    "label": 1  # Vulnerable
                })

            # Add negative examples (simplified: assume non-vulnerable code snippets)
            # Example: Use curated non-vulnerable code from GitHub or SolidiFI
            non_vulnerable = [
                {"text": "function safeTransfer(address to, uint256 value) external { require(to != address(0)); }", "label": 0},
                {"text": "function setOwner(address newOwner) external onlyOwner { owner = newOwner; }", "label": 0}
            ]
            dataset.extend(non_vulnerable)

            print(f"[+] Prepared {len(dataset)} training examples")
        except Exception as e:
            print(f"[-] Training data preparation failed: {e}")
        return dataset

    def fine_tune_llm(self, dataset: List[Dict]):
        """Fine-tune DistilBERT for vulnerability detection."""
        print("[*] Fine-tuning LLM")
        try:
            # Prepare dataset
            df = pd.DataFrame(dataset)
            train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

            def tokenize_function(examples):
                return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

            train_encodings = tokenize_function(train_df.to_dict("records"))
            eval_encodings = tokenize_function(eval_df.to_dict("records"))

            class VulnerabilityDataset(torch.utils.data.Dataset):
                def __init__(self, encodings, labels):
                    self.encodings = encodings
                    self.labels = labels

                def __getitem__(self, idx):
                    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                    item["labels"] = torch.tensor(self.labels[idx])
                    return item

                def __len__(self):
                    return len(self.labels)

            train_dataset = VulnerabilityDataset(train_encodings, train_df["label"].tolist())
            eval_dataset = VulnerabilityDataset(eval_encodings, eval_df["label"].tolist())

            # Training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(self.knowledge_dir, "model"),
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=os.path.join(self.knowledge_dir, "logs"),
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True
            )

            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=DataCollatorWithPadding(self.tokenizer)
            )

            # Train
            trainer.train()
            self.model.save_pretrained(os.path.join(self.knowledge_dir, "fine_tuned_model"))
            self.tokenizer.save_pretrained(os.path.join(self.knowledge_dir, "fine_tuned_model"))
            print("[+] LLM fine-tuning completed")
        except Exception as e:
            print(f"[-] LLM fine-tuning failed: {e}")

    def predict_vulnerabilities(self, code_snippet: str) -> Dict:
        """Predict vulnerabilities in a code snippet using fine-tuned LLM."""
        print(f"[*] Predicting vulnerabilities for snippet: {code_snippet[:50]}...")
        try:
            inputs = self.tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            vuln_prob = probs[0][1].item()
            return {
                "snippet": code_snippet,
                "vulnerability_probability": vuln_prob,
                "is_vulnerable": vuln_prob > 0.5
            }
        except Exception as e:
            print(f"[-] Prediction failed: {e}")
            return {}

    def analyze_contract(self, findings_path: Optional[str] = None, contract_path: Optional[str] = None) -> Optional[Dict]:
        """Analyze findings and contract for patterns and predictions."""
        print("[*] Analyzing contract data")
        result = {"patterns": [], "predictions": []}
        try:
            # Process findings
            if findings_path and os.path.exists(findings_path):
                findings = self.parse_findings(findings_path)
                self.store_findings(findings)
                # Extract patterns (e.g., common functions)
                function_patterns = {}
                for finding in findings:
                    if "function" in finding["location"].lower():
                        func_name = finding["location"].split("(")[0]
                        function_patterns[func_name] = function_patterns.get(func_name, 0) + 1
                result["patterns"] = [
                    {"function": func, "count": count, "swc_ids": [f["swc_id"] for f in findings if func in f["location"]]}
                    for func, count in function_patterns.items()
                ]

            # Process contract source
            if contract_path and os.path.exists(contract_path):
                with open(contract_path, "r") as f:
                    source_code = f.read()
                # Split into functions (simplified regex)
                import re
                functions = re.findall(r"function\s+(\w+)\s*\([^)]*\)\s*(?:public|external|internal|private)?\s*{[^}]*}", source_code)
                for func in functions:
                    func_code = re.search(rf"function\s+{func}\s*\([^)]*\)\s*(?:public|external|internal|private)?\s*{{[^}}]*}}", source_code)
                    if func_code:
                        prediction = self.predict_vulnerabilities(func_code.group(0))
                        result["predictions"].append(prediction)

            # Save results
            identifier = os.path.basename(findings_path or contract_path or "unknown").replace(".json", "").replace(".sol", "")
            output_path = os.path.join(self.knowledge_dir, f"{identifier}_knowledge.json")
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[+] Saved knowledge to {output_path}")
            return result
        except Exception as e:
            print(f"[-] Contract analysis failed: {e}")
            return None

    def run_knowledge_task(self, task: Dict) -> Optional[Dict]:
        """Execute knowledge task from task.yaml."""
        print(f"[*] Running knowledge task: {task.get('id', 'unknown')}")
        action = task.get("action")
        if action != "knowledge":
            print(f"[-] Invalid action for KnowledgeAgent: {action}")
            return None

        findings_path = task.get("findings_path")
        contract_path = task.get("contract_path")
        train = task.get("train", False)

        if not findings_path and not contract_path:
            print("[-] No findings_path or contract_path provided")
            return None

        # Train LLM if requested
        if train:
            dataset = self.prepare_training_data()
            if dataset:
                self.fine_tune_llm(dataset)

        # Analyze findings or contract
        result = self.analyze_contract(findings_path, contract_path)
        if result:
            output_path = os.path.join(self.knowledge_dir, "task_result.json")
            with open(output_path, "w") as f:
                json.dump({"task_id": task.get("id"), "status": "completed", "result": result}, f, indent=2)
            print(f"[+] Saved task result to {output_path}")
        return result

# Example usage (for testing)
if __name__ == "__main__":
    agent = KnowledgeAgent()
    task = {
        "id": 1,
        "action": "knowledge",
        "findings_path": "~/agentic-ai/data/findings/0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9.json",
        "contract_path": "~/agentic-ai/data/contracts/0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9.sol",
        "train": True
    }
    result = agent.run_knowledge_task(task)
    print(json.dumps(result, indent=2))