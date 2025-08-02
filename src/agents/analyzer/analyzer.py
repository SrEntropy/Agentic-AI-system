import os
import json
import subprocess
from typing import Dict, Optional, List
#from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

class AnalyzerAgent:
    def __init__(self, data_dir: str = "~/agentic-ai/data"):
        """Initialize AnalyzerAgent with data directories and embeddings."""
        self.data_dir = os.path.expanduser(data_dir)
        self.contracts_dir = os.path.join(self.data_dir, "contracts")   #Folder for .json and .sol contracts
        self.findings_dir = os.path.join(self.data_dir, "findings")     #Fodler for vulnerable reports
        self.results_dir = os.path.join(self.data_dir, "results")       #Folder for final task results
        os.makedirs(self.findings_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        # Load language model embeddings (used for semmantic indexing, optional for now)
        #self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")
        # Initialize vector store for similarity search (can be disabled if unused)
        """self.vectorstore = Chroma(
            collection_name="vulnerabilities",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(self.data_dir, "chromadb")
        )"""

    def run_mythril(self, contract_data: Dict, sol_path: Optional[str] = None) -> List[Dict]:
        """Run Mythril analysis on bytecode or source code."""
        print(f"[*] Running Mythril for {contract_data.get('contract_name', 'unknown')}")
        findings = []
        try:
            # Determine analysis input: bytecode or .sol file
            if contract_data.get("runtime_bytecode") and contract_data.get("address") != "local":
                print("Running Mythril analysis on runtime bytecode..")
                address = contract_data["address"]
                bytecode = contract_data["runtime_bytecode"].lstrip("0x")
                cmd = ["myth", "analyze", "-c", bytecode, "-a", address, "--execution-timeout", "300"]
            elif sol_path:
                print("Running Mythril analysis on Source Code...")
                cmd = ["myth", "analyze", sol_path, "--execution-timeout", "300"]
            else:
                print("[-] No runtime bytecode or source file for Mythril")
                return findings

            # Run Mythril as subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                try:
                    # Extract vulnerabilities from output (Parse JSON output)
                    output = json.loads(result.stdout)
                    for issue in output.get("issues", []):
                        findings.append({
                            "tool": "Mythril",
                            "severity": issue.get("severity", "Unknown"),
                            "title": issue.get("title", "Unknown issue"),
                            "description": issue.get("description", ""),
                            "location": issue.get("function", "N/A"),
                            "swc_id": issue.get("swc-id", "N/A")
                        })
                    print(f"[+] Mythril found {len(findings)} issues")
                except json.JSONDecodeError:
                    print(f"[-] Mythril output parsing failed: {result.stdout[:100]}...")
            else:
                print(f"[-] Mythril failed: {result.stderr}")
                print("[DEBUG] Mythril STDOUT:\n", result.stdout[:1000])  # Show first 1000 chars
                print("[DEBUG] Mythril STDERR:\n", result.stderr[:1000])  # Show first 1000 chars
        except subprocess.TimeoutExpired:
            print("[-] Mythril timed out after 600 seconds")
        except Exception as e:
            print(f"[-] Mythril error: {e}")
        return findings

    def run_slither(self, sol_path: str) -> List[Dict]:
        """Run Slither analysis on given .sol file/source code and extract issues."""
        print(f"[*] Running Slither for {sol_path}")
        findings = []
        try:
            # Execute Slither with JSON output
            cmd = ["slither", sol_path, "--json", "-"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout)
                    for detector in output.get("results", {}).get("detectors", []):
                        for element in detector.get("elements", []):
                            findings.append({
                                "tool": "Slither",
                                "severity": detector.get("impact", "Unknown"),
                                "title": detector.get("check", "Unknown issue"),
                                "description": detector.get("description", ""),
                                "location": f"{element.get('source_mapping', {}).get('filename_absolute', 'N/A')}:{element.get('source_mapping', {}).get('lines', [])}",
                                "swc_id": detector.get("swc-id", "N/A")
                            })
                    print(f"[+] Slither found {len(findings)} issues")
                except json.JSONDecodeError:
                    print(f"[-] Slither output parsing failed: {result.stdout[:100]}...")
            else:
                print(f"[-] Slither failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("[-] Slither timed out after 300 seconds")
        except Exception as e:
            print(f"[-] Slither error: {e}")
        return findings

    """ def index_findings(self, findings: List[Dict], identifier: str):
        ""Embed and index findings in ChromaDB for (semantic search)used for future search or LLM use.""
        print(f"[*] Indexing findings for {identifier}")
        try:
            documents = [
                f"{finding['title']}: {finding['description']} (Severity: {finding['severity']}, SWC: {finding['swc_id']})"
                for finding in findings
            ]
            if documents:
                self.vectorstore.add_texts(
                    texts=documents,
                    metadatas=[{"identifier": identifier, "tool": f["tool"]} for f in findings]
                )
                print(f"[+] Indexed {len(documents)} findings")
            else:
                print("[-] No findings to index")
        except Exception as e:
            print(f"[-] ChromaDB indexing failed: {e}")"""

    def analyze_contract(self, contract_data: Dict, sol_path: Optional[str] = None) -> Optional[Dict]:
        """Analyze a contract using Mythril and Slither, and store results."""
        identifier = contract_data.get("address", contract_data.get("contract_name", "unknown"))
        print(f"[*] Analyzing contract: {identifier}")
        findings = []

        # Run Mythril if applicable and append results
        mythril_findings = self.run_mythril(contract_data, sol_path)
        findings.extend(mythril_findings)

        # Run Slither if source(.sol file) file exists
        if sol_path and os.path.exists(sol_path):
            slither_findings = self.run_slither(sol_path)
            findings.extend(slither_findings)
        else:
            print(f"[-] No source file for Slither analysis: {sol_path}")

        if not findings:
            print(f"[-] No vulnerabilities found for {identifier}")
            return None

        # Save findings to disk
        findings_path = os.path.join(self.findings_dir, f"{identifier}.json")
        with open(findings_path, "w") as f:
            json.dump(findings, f, indent=2)
        print(f"[+] Saved findings to {findings_path}")

        # Index findings in ChromaDB (optional semantic memory)
        self.index_findings(findings, identifier)

        return {"identifier": identifier, "findings": findings}

    def run_analysis(self, task: Dict) -> Optional[Dict]:
        """Execute analysis task from task.yaml."""
        print(f"[*] Running analysis task: {task.get('id', 'unknown')}")

        action = task.get("action")
        if action != "analyze":
            print(f"[-] Invalid action for AnalyzerAgent: {action}")
            return None

        contract_identifier = task.get("contract_identifier")
        if not contract_identifier:
            print("[-] No contract_identifier provided")
            return None

        # Load contract data metadata and source code paths
        contract_json_path = os.path.join(self.contracts_dir, f"{contract_identifier}.json")
        sol_path = os.path.join(self.contracts_dir, f"{contract_identifier}.sol")

        contract_data = None
        if os.path.exists(contract_json_path):
            with open(contract_json_path, "r") as f:
                contract_data = json.load(f)
        else:
            print(f"[-] Contract data not found: {contract_json_path}")
            contract_data = {"address": contract_identifier, "contract_name": contract_identifier}

        # Determine source path
        if not os.path.exists(sol_path):
            sol_path = None
            print(f"[-] Source file not found: {sol_path}")

        # Run full analysis pipeline
        result = self.analyze_contract(contract_data, sol_path)

        # Save task result or output of (for multi-agent system to consume later)
        if result:
            task_result_path = os.path.join(self.results_dir, "task_result.json")
            with open(task_result_path, "w") as f:
                json.dump({"task_id": task.get("id"), "status": "completed", "result": result}, f, indent=2)
            print(f"[+] Saved task result to {task_result_path}")

        return result

# Example usage (for testing)
if __name__ == "__main__":
    agent = AnalyzerAgent()
    task = {
        "id": 1,
        "action": "analyze",
        "contract_identifier": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9"
    }
    result = agent.run_analysis(task)
    print(json.dumps(result, indent=2))







