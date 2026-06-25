#!/usr/bin/env python3
"""Execute only the standalone focused-analysis cells in the CIFAR notebook."""

import argparse
import json
from pathlib import Path

from jupyter_client import KernelManager


CELL_MARKERS = (
    "# Standalone bootstrap for the focused four-model comparison.",
    "clean_x, clean_y = get_cifar10_test_sample(CLEAN_SAMPLE_INDEX)",
    "# Cell 16 optimization strategy:",
    "def case_activation_frame(record):",
    "plot_joint_case(succeeded_case, 'Attack succeeded on all four models')",
)


def find_cells(notebook):
    selected = []
    for marker in CELL_MARKERS:
        matches = [
            (index, cell)
            for index, cell in enumerate(notebook["cells"])
            if cell.get("cell_type") == "code"
            and "".join(cell.get("source", [])).startswith(marker)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one code cell starting with {marker!r}, found {len(matches)}")
        selected.append(matches[0])
    return selected


def execute_cell(client, source, timeout):
    message_id = client.execute(source, store_history=True, allow_stdin=False)
    outputs = []
    execution_count = None

    while True:
        message = client.get_iopub_msg(timeout=timeout)
        if message.get("parent_header", {}).get("msg_id") != message_id:
            continue
        message_type = message["header"]["msg_type"]
        content = message["content"]

        if message_type == "status" and content.get("execution_state") == "idle":
            break
        if message_type == "execute_input":
            execution_count = content.get("execution_count")
        elif message_type == "stream":
            print(content["text"], end="", flush=True)
            outputs.append(
                {
                    "output_type": "stream",
                    "name": content["name"],
                    "text": content["text"],
                }
            )
        elif message_type in ("display_data", "execute_result"):
            output = {
                "output_type": message_type,
                "data": content.get("data", {}),
                "metadata": content.get("metadata", {}),
            }
            if message_type == "execute_result":
                output["execution_count"] = content.get("execution_count")
            outputs.append(output)
        elif message_type == "error":
            outputs.append(
                {
                    "output_type": "error",
                    "ename": content.get("ename", ""),
                    "evalue": content.get("evalue", ""),
                    "traceback": content.get("traceback", []),
                }
            )
            raise RuntimeError(
                f"Cell execution failed: {content.get('ename')}: {content.get('evalue')}"
            )

    return execution_count, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook",
        default="notebooks/cifar10_latest_robustness_layer_analysis.ipynb",
    )
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    notebook_path = Path(args.notebook).resolve()
    repo_root = notebook_path.parent.parent
    notebook = json.loads(notebook_path.read_text())
    selected = find_cells(notebook)

    manager = KernelManager(kernel_name="python3")
    manager.start_kernel(cwd=str(repo_root))
    client = manager.client()
    client.start_channels()
    client.wait_for_ready(timeout=60)

    try:
        for position, (cell_index, cell) in enumerate(selected, start=1):
            print(f"[{position}/{len(selected)}] executing notebook cell index {cell_index}", flush=True)
            source = "".join(cell["source"])
            execution_count, outputs = execute_cell(client, source, args.timeout)
            cell["execution_count"] = execution_count
            cell["outputs"] = outputs
            notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
            print(f"    saved {len(outputs)} outputs", flush=True)
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)

    print(f"Updated notebook: {notebook_path}")


if __name__ == "__main__":
    main()
