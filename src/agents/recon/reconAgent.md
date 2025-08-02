reconAgent:
Initialization(__init__):
-Loads API keys (ETHERSCAN_API_KEY, optional INFURA_URL)
-Sets up folders for:
-Saved contract JSONs
-Flattened source .sol files
-Immunefi project metadata
-Prepares Web3 connection (optional)
-✅ You’re setting up a structured file system & optionally hooking into Infura for fallback bytecode.


fetch_contract_data(contract_address)
-📡 Pulls contract source, ABI, and runtime bytecode from Etherscan.
-🧰 Parses and extracts verified source tree (multi-file).
-🧪 Tries to recompile the contract with the correct Solidity version (from pragma).
🗃 Saves:
-Full metadata as JSON
-Flattened .sol file
-Creation/runtime bytecode
-✅ It’s your single-contract analyzer. This is used to learn everything you can from an on-chain contract address. First blood, basically.



# fetch_platform_targets(platform="immunefi")
-🌍 Scrapes Immunefi’s explore page.
-🕵️‍♂️ Digs into each bounty’s page to:
-Extract GitHub repos
-Extract contract addresses (via regex)
-🧾 Builds a list of bounty target dicts: name, url, github, contracts
-✅ This is your target scraper. It gives your agent a list of real-world bounty projects to analyze automatically.
