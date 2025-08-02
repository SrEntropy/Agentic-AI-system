

import os
import json
import time
import requests
from web3 import Web3
from bs4 import BeautifulSoup
from git import Repo
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Dict, Optional, List
from dotenv import load_dotenv


class ReconAgent:
    def __init__(self, etherscan_api_key:str,data_dir: str = "~/agentic-ai/data"):
        self.etherscan_api_key = etherscan_api_key
        self.etherscan_url = "https://api.etherscan.io/api"
        self.data_dir = os.path.expanduser(data_dir)
        self.contracts_dir = os.path.join(self.data_dir, "contracts")
        self.projects_dir = os.path.join(self.data_dir, "projects", "immunifi")
        os.makedirs(self.contracts_dir, exist_ok = True)
        os.makedirs(self.projects_dir, exist_ok = True)
        infura_key = os.getenv("INFURA_KEY")
        if not infura_key:
            raise ValueError("INFURA_KEY not set in environment")
        self.w3 = Web3(Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{infura_key}"))
        if not self.w3.is_connected():
            print("[!] Web3 connection failed")


    def fetch_contract_data(self, contract_address: str) -> Optional[Dict]:
        """Fetch contract metadat from Etherscan."""
        try:
            #Checks if address is valid
            if not Web3.is_address(contract_address):
                print(f"Invalid address: {contract_address}")
                return None
            contract_address  = Web3.to_checksum_address(contract_address)

            #Set up API parameters to fetch from etherscan
            params = {
                "module": "contract",
                "action": "getsourcecode",
                "address": contract_address,
                "apikey": self.etherscan_api_key
            }

            #Make API request and response to json format
            response = requests.get(self.etherscan_url, params = params, timeout = 20)
            response.raise_for_status()
            data = response.json() 
            print(f"Source code API response: status={data.get('status')}, message={data.get('message')}")
            #Check API response, 1 in success and 0/empty on failure
            if data["status"] != "1" or not data["result"]:
                print(f"Etherscan Error: {data.get('message', 'No data')}")
                return None
            
            #Extract metadata
            result  = data["result"][0]
            source_code = result.get("SourceCode", "")
            contract_name = result.get("ContractName", "")
            abi = result.get("ABI", "[]")

            #Add delay before second request
            # 0.2 seconds (5 req/sec max for free API)
            time.sleep(0.2)

            #Etherscan API request for Bytecode
            #Update API params
            params["module"] = "proxy"
            params["action"] = "eth_getCode"
            params["tag"] = "latest"
            #params["action"] = "getcontractcode"
            response = requests.get(self.etherscan_url, params = params, timeout = 30)
            response.raise_for_status()
            bytecode_data = response.json()
            print(f"Bytecode API response: status={bytecode_data.get('status')}, message={bytecode_data.get('message')}")

            bytecode = ""
            bytecode = bytecode_data.get("result", "")
            if bytecode:
                print(f"Bytecode fetched from Etherscan: {bytecode[:20]}...")
            else:
                print("Bytecode field empty in Etherscan response")

            
            #Fallback to Web3 for Bytecode
            #If no bytcode from API,
            # it connects to fetch deployed contract's bytecode from blockchain 
            if not bytecode and self.w3.is_connected():
                try:
                    bytecode = self.w3.eth.get_code(contract_address).hex()
                    print(f"Bytecode fetched from Infura: {bytecode[:20]}...")
                except Exception as e:
                    print(f"Infura bytecode fetch failed: {e}")
            #Construct the contract Data: dictionary
            contract_data = {
                "address": contract_address,
                "contract_name": contract_name,
                "source_code": source_code,
                "abi": json.loads(abi) if abi else [],
                "bytecode": bytecode
            }

            #Save to File and returns data
            output_path = os.path.join(self.contracts_dir, f"{contract_address}.json")
            with open(output_path, "w") as f:
                json.dump(contract_data, f, indent = 2)
            print(f"Saved contract data to {output_path}")
            return contract_data

        except Exception as e:
            print(f"Error fetching contract: {e}")
            return None
        

    def fetch_platform_targets(self, platform: str = "immunefi") -> List[Dict]:
        """Scrape bounty targets from Immunefi."""
        if platform.lower() != "immunefi":
            raise NotImplementedError("Only Immunefi supported")
        
        print("[*] Scraping Immunefi...")
        url = "https://immunefi.com/explore/"
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            response = requests.get(url, headers = headers, timeout = 30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return []
        time.sleep(1)
        soup = BeautifulSoup(response.text, "html.parser")
        cards = soup.find_all("a", href = True)
        targets = []

        for card in cards:
            href = card["href"]
            if "/bounty/" in href:
                try:
                    name = href.split("/")[-2]
                    target_url = f"https://immunefi.com{href}"
                #Fetch bounty page for Github link
                    bounty_response = requests.get(target_url, headers = headers, timeout = 30)
                    bounty_response.raise_for_status()
                    bounty_soup = BeautifulSoup(bounty_response.text, "html.parser")
                    github_link = bounty_soup.find("a", href = lambda x: x and "github.com" in x)
                    github_url = github_link["href"] if github_link else None
                    targets.append({
                    "name": name,
                    "url": target_url,
                    "github": github_url
                })
                    time.sleep(1) # Avoid rate limiting
                except requests.RequestException as e:
                    print(f"Error fetching {target_url}: {e}")
                    continue

        print(f"[+] Found {len(targets)} targets")
        return targets
        
    def process_target(self, target: Dict) -> None:
        """Clone repo and index Solidity files."""
        project_dir = os.path.join(self.projects_dir, target["name"])
        os.makedirs(project_dir, exist_ok = True)

        #Save metadata
        metadata_path = os.path.join(project_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(target, f, indent = 2)
        
        #Clone repo
        if target.get("github"):
            repo_path = os.path.join(project_dir, "repo")
            if not os.path.exists(repo_path):
                try:
                    print(f"[+] Cloning {target['github']}...")
                    Repo.clone_from(target["github"], repo_path)
                except Exception as e:
                    print(f"[!] Failed to clone {target['github']}: {e}")
            
            #Index Solidity files
            self.index_code(repo_path, project_dir)


    def index_code(self, repo_path: str, project_dir: str) -> None:
        """Index Solidity file in ChromaDB."""
        docs = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".sol"):
                    with open(os.path.join(root,file), "r", encoding = "utf-8") as f:
                        docs.append(f.read())
        
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

def run_recon(task: Dict) -> Optional[Dict]:
    """Run ReconAgent based on task configuration"""
    load_dotenv()
    etherscan_api_key = os.environ.get("ETHERSCAN_API_KEY")
    if not etherscan_api_key:
        print("ETHERSCAN_API_KEY not set")
        return None
        
    agent = ReconAgent(etherscan_api_key)
    print("etherscan API key loaded")

    if task.get("action") == "scan" and task.get("contract_address"):
        return agent.fetch_contract_data(task["contract_address"])
    elif task.get("action") == "platform_scan":
        targets = agent.fetch_platform_targets(platform = "immunifi")
        for target in targets:
            agent.process_target(target)
        return {"targets_processed": len(targets)}
    else:
        print("Invalid task action")
        return None









































