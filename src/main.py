import yaml
import json
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Optional
from agents.recon.recon3 import run_recon
from agents.analyzer.analyzer import analyze_contract

class AgentState(TypedDict):
    task: Dict
    contract_data: Optional[Dict]
    findings: Optional[Dict]

def recon_node(state: AgentState) -> AgentState:
    result = run_recon(state["task"])
    state["contract_data"] = result
    return state

def analyzer_node(state: AgentState) -> AgentState:
 """source_path = None
    if state["contract_data"] and isinstance(state["contract_data"], dict) and state["contract_data"].get("source_code"):
        contract_address = state["task"]["contract_address"]
        source_path = os.path.join("~/agentic-ai/data/contracts", f"{contract_address}.sol")
        source_path = os.path.expanduser(source_path)
        os.makedirs(os.path.dirname(source_path), exist_ok=True)
        with open(source_path, "w") as f:
            f.write(state["contract_data"]["source_code"])
    elif state["task"].get("target"):
        source_path = os.path.expanduser(state["task"]["source_code"])


    if source_path and os.path.exists(source_path):
        print("findings: ", source_path)
        findings = analyze_contract(source_path)
        state["findings"] = findings
    else:
        state["findings"] = {"error": "No source code available"}
    return state    """
 return state

workflow = StateGraph(AgentState)
workflow.add_node("recon", recon_node)
#workflow.add_node("analyzer", analyzer_node)
#workflow.add_edge("recon", "analyzer")
workflow.add_edge("recon", END)
#workflow.add_edge("analyzer", END)
workflow.set_entry_point("recon")
graph = workflow.compile()

def main():
    task_path = os.path.expanduser("~/agentic-ai/data/task.yaml")
    try:
        with open(task_path) as f:
            config = yaml.safe_load(f)
            tasks = config.get("tasks", [])
            if not tasks:
                print("Error: NO tasks found in task.yaml")
                return
    
            for task in tasks:
                if not isinstance(task, dict) or "action" not in task:
                        print(f"Invalid task format: {task}")
                        continue
                state = {"task": task, "contract_data": None, "findings": None}
                print(f"Processing task ID: {task.get('id', 'unknown')}")
                result = graph.invoke(state)
                output_path = os.path.expanduser("~/agentic-ai/data/task_result.json")
                with open(output_path, "w") as out_f:
                    json.dump(result, out_f, indent=2)
                print(f"Task result saved to {output_path}")
                print(json.dumps(result, indent=2))
    except FileNotFoundError:
        print(f"Error:{task_path} not found")
    except yaml.YAMLError as e:
        print(f"Error parsing task.yaml: {e}")

if __name__ == "__main__":
    main()