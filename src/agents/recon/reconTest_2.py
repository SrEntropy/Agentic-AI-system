import os
import yaml
from recon2 import run_recon
from dotenv import load_dotenv

load_dotenv()

def test_valid_contract():
    print("\n[TEST T1] Valid contract address scan:")
    task = {
        "action": "scan",
        "contract_address": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9"  # Aave V2
    }
    result = run_recon(task)
    print("✅ Result:", result is not None)

def test_invalid_contract():
    print("\n[TEST T2] Invalid contract address scan:")
    task = {
        "action": "scan",
        "contract_address": "0x1234567890INVALID"
    }
    result = run_recon(task)
    print("✅ Error handled:", result is None)

def test_platform_scan():
    print("\n[TEST T3] Immunefi platform scan:")
    task = {
        "action": "platform_scan",
        "platform": "immunefi"
    }
    result = run_recon(task)
    print(f"✅ Targets processed: {result.get('targets_processed', 0)}" if result else "❌ Platform scan failed")

def test_local_sol_analysis():
    print("\n[TEST T4] Analyze local .sol file:")
    sol_file = "~/agentic-ai/data/web3/defivulnlabs/Reentrancy.sol"
    task = {
        "action": "scan",
        "target": sol_file
    }
    result = run_recon(task)
    print("✅ Local analysis result:", result is not None)

def test_from_task_yaml(path="~/agentic-ai/data/task.yaml"):
    print("\n[TEST T5] Running from task.yaml:")
    try:
        path = os.path.expanduser(path)
        with open(path, "r") as f:
            task_data = yaml.safe_load(f)

        if isinstance(task_data, dict):
            task_data = [task_data]

        for task in task_data:
            print(f"\n➡️ Running task: {task}")
            result = run_recon(task)
            print("✅ Task result:", result is not None)
    except Exception as e:
        print(f"❌ Failed to load or run task.yaml: {e}")

if __name__ == "__main__":
    print("=== ReconAgent Functional Test Suite ===")
    test_valid_contract()
    test_invalid_contract()
    test_platform_scan()
    test_local_sol_analysis()
    test_from_task_yaml()
