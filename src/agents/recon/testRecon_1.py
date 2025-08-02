import os
import yaml
from recon import run_recon
from dotenv import dotenv_values

def force_reload_dotenv():
    values = dotenv_values()  # defaults to .env in current dir
    for k, v in values.items():
        os.environ[k] = v
    print("[*] Reloaded .env into os.environ.")

def test_valid_contract():
    print("\n[TEST] Valid contract address scan:")
    task = {
        "action": "scan",
        "contract_address": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9"  # Aave V2 LendingPool
    }
    result = run_recon(task)
    print("✅ Result:", result is not None)

def test_invalid_contract():
    print("\n[TEST] Invalid contract address scan:")
    task = {
        "action": "scan",
        "contract_address": "0x1234567890INVALID"
    }
    result = run_recon(task)
    print("✅ Error handled:", result is None)

def test_missing_api_key():
    print("\n[TEST] Missing ETHERSCAN_API_KEY and INFURA_KEY:")
    # Temporarily unset env vars
    os.environ.pop("ETHERSCAN_API_KEY", None)
    os.environ.pop("INFURA_KEY", None)

    try:
        task = {
            "action": "scan",
            "contract_address": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9"
        }
        result = run_recon(task)
        print("✅ Error handled:", result is None)
    except Exception as e:
        print(f"⚠️ Exception raised: {e}")

    # Restore vars after test
    force_reload_dotenv()

def test_platform_scan():
    force_reload_dotenv()

    print("\n[TEST] Immunefi platform scan:")
    task = {
        "action": "platform_scan"
    }
    result = run_recon(task)
    print(f"✅ Targets processed: {result.get('targets_processed', 0)}" if result else "❌ Platform scan failed")

def test_from_task_yaml(path="~/agentic-ai/data/task.yaml"):
    print("\n[TEST] Running from task.yaml:")
    try:
        path = os.path.expanduser(path)
        with open(path, "r") as f:
            task_data = yaml.safe_load(f)

        if isinstance(task_data, dict):
            task_data = [task_data]  # Wrap in list for single-task files

        for task in task_data:
            print(f"\n➡️ Running task: {task}")
            result = run_recon(task)
            print("✅ Result:", result is not None)

    except Exception as e:
        print(f"❌ Failed to load or run task.yaml: {e}")


if __name__ == "__main__":
    print("=== ReconAgent Test Suite ===")

    # Test each edge case
    test_valid_contract()
    test_invalid_contract()
    test_missing_api_key()
    force_reload_dotenv()
    test_platform_scan()        
    force_reload_dotenv()
    test_from_task_yaml()

