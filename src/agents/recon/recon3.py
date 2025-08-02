
import os
import json
import requests
import time
import subprocess
import solcx
import re
from web3 import Web3
from bs4 import BeautifulSoup
from git import Repo
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from chromadb import PersistentClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Dict, Optional, List
from dotenv import load_dotenv

load_dotenv()
etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
infura_url = os.getenv("INFURA_API_KEY")
print("key: ", infura_url)
if not etherscan_api_key:
    raise ValueError("ETHERSCAN_API_KEY not set")

class ReconAgent:
    def __init__(self, etherscan_api_key: str, infura_url: str = None, data_dir: str = "~/agentic-ai/data"):
        self.etherscan_api_key = etherscan_api_key
        self.etherscan_url = "https://api.etherscan.io/api"
        self.data_dir = os.path.expanduser(data_dir)
        self.contracts_dir = os.path.join(self.data_dir, "contracts")
        self.projects_dir = os.path.join(self.data_dir, "projects", "immunefi")  # Fixed typo from Version B
        os.makedirs(self.contracts_dir, exist_ok=True)
        os.makedirs(self.projects_dir, exist_ok=True)
        self.w3 = Web3(Web3.HTTPProvider(infura_url)) if infura_url else None
        # Note: ChromaDB initialized per method to reduce memory; add to __init__ for persistent DB later

    def fetch_contract_data(self, contract_address: str) -> Optional[Dict]:
        """Fetch contract metadata from Etherscan and compile creation bytecode if all sources are complete."""
        print(f"[*] Fetching data for contract: {contract_address}")
        try:
            if not Web3.is_address(contract_address):
                print(f"[-] Invalid address: {contract_address}")
                return None
            contract_address = Web3.to_checksum_address(contract_address)

            # Fetch source code and ABI with retries
            params = {
                "module": "contract",
                "action": "getsourcecode",
                "address": contract_address,
                "apikey": self.etherscan_api_key
            }
            for attempt in range(3):
                try:
                    response = requests.get(self.etherscan_url, params=params, timeout=10)
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    print(f"[-] Source code attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        print(f"[-] Error fetching source code after 3 attempts: {e}")
                        return None
            data = response.json()
            print(f"[+] Source code API response: status={data.get('status')}, message={data.get('message')}")

            if data["status"] != "1" or not data["result"]:
                print(f"[-] Etherscan source code error: {data.get('result', 'No data')}")
                return None

            result = data["result"][0]
            source_code = result.get("SourceCode", "")
            contract_name = result.get("ContractName", "")
            abi = result.get("ABI", "[]")
            etherscan_compiler_version = result.get("CompilerVersion", "").lstrip("v")  # e.g., "0.6.12+commit.27d51765"

            # Fetch runtime bytecode with retries
            time.sleep(0.2)  # Respect Etherscan rate limit
            params = {
                "module": "proxy",
                "action": "eth_getCode",
                "address": contract_address,
                "tag": "latest",
                "apikey": self.etherscan_api_key
            }
            for attempt in range(3):
                try:
                    response = requests.get(self.etherscan_url, params=params, timeout=10)
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    print(f"[-] Bytecode attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        print(f"[-] Error fetching bytecode after 3 attempts: {e}")
                        return None
            bytecode_data = response.json()
            print(f"[+] Bytecode result length: {len(bytecode_data.get('result', ''))}")

            bytecode = ""
            if bytecode_data.get("result"):
                bytecode = bytecode_data["result"]
                if bytecode.startswith("0x"):
                    print(f"[+] Runtime bytecode fetched from Etherscan: {bytecode[:20]}...")
                else:
                    print("[-] Invalid bytecode format from eth_getCode")

            if not bytecode and self.w3 and self.w3.is_connected():
                try:
                    bytecode = self.w3.eth.get_code(contract_address).hex()
                    print(f"[+] Runtime bytecode fetched from Infura: {bytecode[:20]}...")
                except Exception as e:
                    print(f"[-] Infura bytecode fetch failed: {e}")

            # Parse and validate source code
            source_data = None
            if source_code:
                try:
                    cleaned_source = source_code.strip()
                    if cleaned_source.startswith("{{") and cleaned_source.endswith("}}"):
                        cleaned_source = cleaned_source[1:-1]
                    if cleaned_source.startswith("{"):
                        source_data = json.loads(cleaned_source)
                    else:
                        source_data = {"sources": {f"{contract_name}.sol": {"content": cleaned_source}}}
                except Exception as e:
                    print(f"[-] Source code parsing failed: {e}")
                    print(f"[-] Source code preview: {source_code[:100]}...")
                    source_data = None

            # Create and validate source files
            creation_bytecode = ""
            compiler_version = None
            missing_files = False
            if source_data and "sources" in source_data:
                try:
                    source_dir = os.path.join(self.contracts_dir, f"{contract_address}_source")
                    os.makedirs(source_dir, exist_ok=True)
                    source_files = []

                    for path, content in source_data["sources"].items():
                        if "content" not in content or not content["content"].strip():
                            print(f"[-] Missing or empty content for {path}")
                            missing_files = True
                            continue
                        full_path = os.path.join(source_dir, path)
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        with open(full_path, "w") as f:
                            f.write(content["content"])
                        source_files.append(full_path)

                    if not missing_files and source_files:
                        # Extract pragma from main contract
                        main_contract_path = next(
                            (p for p in source_files if "InitializableImmutableAdminUpgradeabilityProxy.sol" in p or p.endswith(f"{contract_name}.sol")),
                            source_files[0]
                        )
                        with open(main_contract_path, "r") as f:
                            source_text = f.read()
                        match = re.search(r'pragma solidity\s+(\^?\d+\.\d+\.\d+);', source_text)
                        pragma_version = match.group(1) if match else "0.8.20"

                        # Validate against Etherscan's CompilerVersion
                        compiler_version = etherscan_compiler_version.split("+")[0] if etherscan_compiler_version else pragma_version
                        if pragma_version.lstrip("^") != compiler_version and "^" not in pragma_version:
                            print(f"[!] Warning: Pragma version {pragma_version} differs from Etherscan {compiler_version}")

                        try:
                            solcx.install_solc(compiler_version)
                            solcx.set_solc_version(compiler_version)
                            compiled = solcx.compile_files(
                                source_files,
                                base_path=source_dir,
                                output_values=["bin"],
                                optimize=True,
                                optimize_runs=200
                            )
                            # Find creation bytecode for the main contract
                            for key in compiled:
                                if key.endswith(f":{contract_name}"):
                                    creation_bytecode = compiled[key]["bin"]
                                    print(f"[+] Creation bytecode compiled with solc {compiler_version}: {creation_bytecode[:20]}...")
                                    break
                            if not creation_bytecode:
                                print("[-] Could not find creation bytecode for main contract")
                        except Exception as e:
                            print(f"[-] solc compilation failed with version {compiler_version}: {e}")
                    elif missing_files:
                        print("[-] Skipping compilation: incomplete source tree (missing imports)")
                except Exception as e:
                    print(f"[-] Creation bytecode compilation failed: {e}")
            else:
                print("[-] Skipping creation bytecode compilation due to invalid or missing source_data")

            if not bytecode:
                print("[-] Warning: No runtime bytecode fetched from Etherscan or Infura")

            # Save output
            contract_data = {
                "address": contract_address,
                "contract_name": contract_name,
                "source_code": source_code,
                "abi": json.loads(abi) if abi else [],
                "runtime_bytecode": bytecode,
                "creation_bytecode": creation_bytecode
            }

            output_path = os.path.join(self.contracts_dir, f"{contract_address}.json")
            with open(output_path, "w") as f:
                json.dump(contract_data, f, indent=2)
            print(f"[+] Saved contract data to {output_path}")

            if source_data and "sources" in source_data:
                sol_file = os.path.join(self.contracts_dir, f"{contract_address}.sol")
                with open(sol_file, "w") as f:
                    for path, content in source_data["sources"].items():
                        if "content" in content:
                            f.write(content["content"] + "\n")
                print(f"[+] Saved flattened source to {sol_file}")
            else:
                print("[-] Skipping flattened source output due to invalid source_data")

            return contract_data

        except Exception as e:
            print(f"[-] Unexpected error fetching contract data: {e}")
            return None
    
    def fetch_platform_targets(self, platform: str = "immunefi") -> List[Dict]:
        """Scrape bounty targets from Immunefi."""
        print("fetch_platform:", platform)
        if platform.lower() != "immunefi":
            raise NotImplementedError("Only Immunefi supported")
        
        print("[*] Scraping Immunefi...")
        url = "https://immunefi.com/explore/"
        headers = {"User-Agent": "Mozilla/5.0"}
        targets = []
        seen_names = set()

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            cards = soup.find_all("a", href=True)

            for card in cards:
                href = card["href"]
                if "/bounty/" in href:
                    try:
                        name = href.split("/")[-2]
                        if name in seen_names:
                            continue
                        seen_names.add(name)

                        target_url = f"https://immunefi.com{href}"
                        bounty_response = requests.get(target_url, headers=headers, timeout=10)
                        bounty_response.raise_for_status()
                        bounty_soup = BeautifulSoup(bounty_response.text, "html.parser")

                        github_link = bounty_soup.find("a", href=lambda x: x and "github.com" in x)
                        raw_github = github_link["href"] if github_link else None
                        github_url = None

                        if raw_github:
                            github_url = raw_github.split("?")[0].split("/tree")[0].split("/blob")[0]
                            if not github_url.endswith(".git") and github_url.count("/") < 5:
                                # Auto append .git if missing and format is clean
                                github_url = github_url.rstrip("/") + ".git"
                            if github_url.count("/") < 4:  # Still not valid
                                print(f"[-] Skipping non-repo GitHub URL: {raw_github}")
                                github_url = None

                        # Extract contract addresses from bounty page (regex)
                        addresses = re.findall(r"0x[a-fA-F0-9]{40}", bounty_response.text)

                        targets.append({
                            "name": name,
                            "url": target_url,
                            "github": github_url,
                            "contracts": addresses
                        })
                        print(f"[+] Found target: {name}")
                        time.sleep(1)  # Avoid rate limiting
                    except requests.RequestException as e:
                        print(f"[-] Error fetching bounty {href}: {e}")
                        continue

            print(f"[+] Found {len(targets)} valid targets")
            return targets

        except requests.exceptions.RequestException as e:
            print(f"[-] Error fetching {url}: {e}")
            return []
        
    def process_target(self, target: Dict) -> None:
        """Clone repo and index Solidity files."""
        project_dir = os.path.join(self.projects_dir, target["name"])
        os.makedirs(project_dir, exist_ok=True)

        # Save metadata
        metadata_path = os.path.join(project_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(target, f, indent=2)
        print(f"[+] Saved metadata to {metadata_path}")

        # Clone repo
        if target.get("github"):
            repo_path = os.path.join(project_dir, "repo")
            if not os.path.exists(repo_path):
                try:
                    print(f"[+] Cloning {target['github']}...")
                    Repo.clone_from(target["github"], repo_path)
                except Exception as e:
                    print(f"[-] Failed to clone {target['github']}: {e}")

            # Index Solidity files
            self.index_code(repo_path, project_dir)

    def index_code(self, repo_path: str, project_dir: str) -> None:
        """Index Solidity file in ChromaDB."""
        docs = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".sol"):
                    with open(os.path.join(root,file), "r", encoding = "utf-8") as f:
                        docs.append(f.read())
        """
        if not docs:
            print(f"[-] No Solidity files found in {repo_path}")
            return
  
        print(f"[+] Indexing {len(docs)} files...")
        embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-V2")
        vectordb  = Chroma.from_texts(
            docs,
            embeddings = embeddings,
            persist_directory = os.path.join(project_dir, "index")
        )
        vectordb.persist()
        """
    def analyze_local_contract(self, sol_path: str) -> Optional[Dict]:
        """Analyze a local .sol file."""
        print(f"[*] Analyzing local contract: {sol_path}")
        try:
            sol_path = os.path.expanduser(sol_path)
            if not os.path.exists(sol_path):
                print(f"[-] File not found: {sol_path}")
                return None

            with open(sol_path, "r") as f:
                source_code = f.read()

            # Compile with solc
           # Automatically install and set correct solc version
            pragma_version = re.search(r"pragma solidity\s+([^\s;]+)", source_code)
            if pragma_version:
                version = pragma_version.group(1).replace("^", "").replace(">", "")
                solcx.install_solc(version)
                solcx.set_solc_version(version)

            solc_cmd = [
                solcx.get_executable(),
                main_contract_path,
                "--combined-json", "bin",
                "--optimize", "--optimize-runs=200",
                "--base-path", source_dir,
                "--allow-paths", ".", source_dir, os.path.join(source_dir, "contracts")
            ]
            print("[DEBUG] solc_cmd:\n", " ".join(solc_cmd))

            result = subprocess.run(solc_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[-] solc compilation failed: {result.stderr}")
                return None

            # Extract bytecode and ABI
            lines = result.stdout.split("\n")
            creation_bytecode = ""
            abi = ""
            for i, line in enumerate(lines):
                if line.startswith("======= "):
                    if "Binary:" in lines[i + 1]:
                        creation_bytecode = lines[i + 2]
                    if "Contract JSON ABI" in lines[i + 1]:
                        abi = lines[i + 2]

            contract_data = {
                "address": "local",
                "contract_name": os.path.basename(sol_path).replace(".sol", ""),
                "source_code": source_code,
                "abi": json.loads(abi) if abi else [],
                "runtime_bytecode": "",  # Note: Runtime bytecode requires deployment simulation
                "creation_bytecode": creation_bytecode
            }

            output_path = os.path.join(self.contracts_dir, f"{contract_data['contract_name']}.json")
            with open(output_path, "w") as f:
                json.dump(contract_data, f, indent=2)
            print(f"[+] Saved local contract data to {output_path}")

            return contract_data

        except Exception as e:
            print(f"[-] Local contract analysis failed: {e}")
            return None    

def run_recon(task: Dict) -> Optional[Dict]:
    if not etherscan_api_key:
            print("[-] ETHERSCAN_API_KEY not set")
            return None

    agent = ReconAgent(etherscan_api_key, infura_url)
    print("[+] Etherscan API key loaded")

    if task.get("action") == "scan" and task.get("contract_address"):
        return agent.fetch_contract_data(task["contract_address"])
    elif task.get("action") == "scan_platform":
        print("scanning Immunefi.............")
        targets = agent.fetch_platform_targets(platform="immunefi")
        for target in targets:
            agent.process_target(target)
        return {"targets_processed": len(targets)}
    else:
        print("[-] Invalid task action")
        return None


