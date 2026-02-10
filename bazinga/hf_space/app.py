#!/usr/bin/env python3
"""
BAZINGA Network Dashboard
Hugging Face Spaces Deployment

A decentralized federated learning network with Darmiyan Blockchain.
Validation through understanding, not computation.
"""

import gradio as gr
import hashlib
import time
import json
import os
from datetime import datetime
from typing import List, Dict

# Constants
PHI = 1.618033988749895
PHI_4 = PHI ** 4  # 6.854101966
ABHI_AMU = 515
ALPHA_INV = 137

# Simulated network state (in production, this would be persistent)
network_state = {
    "nodes": {},
    "chain": [
        {
            "index": 0,
            "timestamp": time.time(),
            "data": "Genesis Block - Darmiyan Blockchain",
            "previous_hash": "0" * 64,
            "hash": hashlib.sha3_256(b"Genesis Block - Darmiyan Blockchain").hexdigest(),
            "pob_proof": {"P": PHI_4, "G": 1.0, "ratio": PHI_4, "valid": True}
        }
    ],
    "total_pob_proofs": 1,
    "credits": {}
}

def calculate_pob(data: str) -> dict:
    """Calculate Proof-of-Boundary."""
    h = hashlib.sha3_256(data.encode()).digest()
    P = sum(h[:16]) / 256
    G = sum(h[16:]) / 256

    if G == 0:
        G = 0.001

    ratio = P / G
    target = PHI_4
    tolerance = 0.5

    valid = abs(ratio - target) < tolerance

    return {
        "P": round(P, 6),
        "G": round(G, 6),
        "ratio": round(ratio, 6),
        "target": round(target, 6),
        "valid": valid
    }

def get_network_stats():
    """Get current network statistics."""
    active_nodes = len([n for n in network_state["nodes"].values()
                       if time.time() - n.get("last_seen", 0) < 300])

    total_credits = sum(network_state["credits"].values())

    # Calculate network coherence
    if network_state["chain"]:
        valid_proofs = sum(1 for b in network_state["chain"] if b.get("pob_proof", {}).get("valid", False))
        coherence = valid_proofs / len(network_state["chain"])
    else:
        coherence = 0

    return {
        "active_nodes": active_nodes,
        "total_nodes": len(network_state["nodes"]),
        "chain_height": len(network_state["chain"]),
        "total_pob_proofs": network_state["total_pob_proofs"],
        "network_coherence": round(coherence, 3),
        "total_credits": round(total_credits, 2)
    }

def register_node(node_name: str):
    """Register a new node to the network."""
    if not node_name or len(node_name) < 2:
        return "Error: Please enter a valid node name (at least 2 characters)"

    node_id = hashlib.sha256(f"{node_name}:{time.time()}".encode()).hexdigest()[:16]

    network_state["nodes"][node_id] = {
        "name": node_name,
        "joined": time.time(),
        "last_seen": time.time(),
        "pob_count": 0
    }

    network_state["credits"][node_id] = 1.0  # Starting credit

    stats = get_network_stats()

    return f"""
## Node Registered Successfully!

**Node ID:** `{node_id}`
**Name:** {node_name}
**Starting Credits:** 1.0

### To run BAZINGA on your machine:

```bash
pip install bazinga

# Start as a node
bazinga --node --id {node_id}

# Or join the network
bazinga --join
```

### Network Status
- Active Nodes: {stats['active_nodes']}
- Chain Height: {stats['chain_height']}
- Network Coherence: {stats['network_coherence']:.1%}
"""

def submit_pob_proof(node_id: str, data: str):
    """Submit a Proof-of-Boundary."""
    if not node_id or node_id not in network_state["nodes"]:
        return "Error: Invalid node ID. Please register first."

    if not data:
        return "Error: Please enter data to prove."

    # Calculate PoB
    pob = calculate_pob(data)

    # Update node
    network_state["nodes"][node_id]["last_seen"] = time.time()
    network_state["nodes"][node_id]["pob_count"] += 1
    network_state["total_pob_proofs"] += 1

    if pob["valid"]:
        # Award credits
        network_state["credits"][node_id] = network_state["credits"].get(node_id, 0) + 1.0

        # Add to chain
        prev_block = network_state["chain"][-1]
        new_block = {
            "index": len(network_state["chain"]),
            "timestamp": time.time(),
            "data": data[:100],  # Truncate for display
            "node_id": node_id,
            "previous_hash": prev_block["hash"],
            "hash": hashlib.sha3_256(f"{prev_block['hash']}:{data}".encode()).hexdigest(),
            "pob_proof": pob
        }
        network_state["chain"].append(new_block)

        status = "VALID - Block added to chain!"
        credits = network_state["credits"][node_id]
    else:
        status = "INVALID - Ratio not within tolerance"
        credits = network_state["credits"].get(node_id, 0)

    return f"""
## Proof-of-Boundary Result

**Status:** {status}

### PoB Metrics
| Metric | Value |
|--------|-------|
| P (Perception) | {pob['P']} |
| G (Grounding) | {pob['G']} |
| P/G Ratio | {pob['ratio']} |
| Target (φ⁴) | {pob['target']} |
| Valid | {'Yes' if pob['valid'] else 'No'} |

### Your Stats
- Node ID: `{node_id}`
- Total PoB Proofs: {network_state['nodes'][node_id]['pob_count']}
- Credits: {credits}
"""

def view_chain():
    """View the blockchain."""
    if not network_state["chain"]:
        return "Chain is empty."

    output = "## Darmiyan Blockchain\n\n"
    output += f"**Chain Height:** {len(network_state['chain'])} blocks\n\n"

    # Show last 10 blocks
    for block in network_state["chain"][-10:]:
        valid_emoji = "✓" if block.get("pob_proof", {}).get("valid", False) else "✗"
        timestamp = datetime.fromtimestamp(block["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")

        output += f"""
### Block #{block['index']} {valid_emoji}
- **Time:** {timestamp}
- **Hash:** `{block['hash'][:32]}...`
- **Data:** {block.get('data', 'Genesis')[:50]}...
- **PoB Valid:** {block.get('pob_proof', {}).get('valid', 'N/A')}
---
"""

    return output

def view_nodes():
    """View registered nodes."""
    stats = get_network_stats()

    output = f"""## Network Nodes

**Active:** {stats['active_nodes']} | **Total:** {stats['total_nodes']} | **Coherence:** {stats['network_coherence']:.1%}

| Node ID | Name | PoB Count | Credits | Status |
|---------|------|-----------|---------|--------|
"""

    for node_id, node in network_state["nodes"].items():
        is_active = time.time() - node.get("last_seen", 0) < 300
        status = "Active" if is_active else "Inactive"
        credits = network_state["credits"].get(node_id, 0)

        output += f"| `{node_id[:8]}...` | {node['name']} | {node['pob_count']} | {credits:.2f} | {status} |\n"

    if not network_state["nodes"]:
        output += "| - | No nodes registered | - | - | - |\n"

    return output

def get_dashboard():
    """Get main dashboard view."""
    stats = get_network_stats()

    return f"""
# BAZINGA Network

> **Validation through understanding, not computation.**

## Live Statistics

| Metric | Value |
|--------|-------|
| Active Nodes | {stats['active_nodes']} |
| Chain Height | {stats['chain_height']} blocks |
| Total PoB Proofs | {stats['total_pob_proofs']} |
| Network φ-Coherence | {stats['network_coherence']:.1%} |
| Total Credits | {stats['total_credits']:.2f} |

## Core Constants

| Symbol | Name | Value |
|--------|------|-------|
| φ | Golden Ratio | {PHI} |
| φ⁴ | PoB Target | {PHI_4:.6f} |
| ABHI_AMU | Identity Constant | {ABHI_AMU} |
| α⁻¹ | Fine Structure Inverse | {ALPHA_INV} |

## Quick Start

```bash
# Install BAZINGA
pip install bazinga

# Start a node
bazinga --node

# Join the network
bazinga --join

# Mine a block (zero energy)
bazinga --mine

# View chain
bazinga --chain
```

---
*Darmiyan Blockchain: Where meaning validates truth.*
"""

# Create Gradio Interface
with gr.Blocks(title="BAZINGA Network") as demo:
    gr.Markdown("""
    # BAZINGA: Decentralized Federated Learning
    ### Powered by Darmiyan Blockchain & Proof-of-Boundary
    """)

    with gr.Tabs():
        # Dashboard Tab
        with gr.TabItem("Dashboard"):
            dashboard_output = gr.Markdown(get_dashboard())
            refresh_btn = gr.Button("Refresh Dashboard")
            refresh_btn.click(fn=get_dashboard, outputs=dashboard_output)

        # Join Network Tab
        with gr.TabItem("Join Network"):
            gr.Markdown("## Register Your Node")
            node_name_input = gr.Textbox(label="Node Name", placeholder="e.g., my-laptop")
            register_btn = gr.Button("Register Node", variant="primary")
            register_output = gr.Markdown()
            register_btn.click(fn=register_node, inputs=node_name_input, outputs=register_output)

        # Submit PoB Tab
        with gr.TabItem("Submit PoB"):
            gr.Markdown("## Submit Proof-of-Boundary")
            pob_node_id = gr.Textbox(label="Your Node ID", placeholder="Enter your node ID from registration")
            pob_data = gr.Textbox(label="Data to Prove", placeholder="Enter any data...", lines=3)
            pob_btn = gr.Button("Submit Proof", variant="primary")
            pob_output = gr.Markdown()
            pob_btn.click(fn=submit_pob_proof, inputs=[pob_node_id, pob_data], outputs=pob_output)

        # View Chain Tab
        with gr.TabItem("Blockchain"):
            chain_output = gr.Markdown(view_chain())
            chain_refresh = gr.Button("Refresh Chain")
            chain_refresh.click(fn=view_chain, outputs=chain_output)

        # View Nodes Tab
        with gr.TabItem("Nodes"):
            nodes_output = gr.Markdown(view_nodes())
            nodes_refresh = gr.Button("Refresh Nodes")
            nodes_refresh.click(fn=view_nodes, outputs=nodes_output)

        # About Tab
        with gr.TabItem("About"):
            gr.Markdown("""
## About BAZINGA

**BAZINGA** is a decentralized federated learning framework powered by the **Darmiyan Blockchain**.

### Key Features

- **Proof-of-Boundary (PoB):** Validates through golden ratio (φ⁴ ≈ 6.854), not computational puzzles
- **Triadic Consensus:** 3 nodes must understand and agree
- **Zero Energy Mining:** No wasted computation
- **φ-Coherence Filter:** Rejects noise (threshold: 0.618)
- **Credit Economics:** Understanding = currency

### The 5 Integration Layers

1. **Trust Oracle** - φ-weighted reputation with time decay
2. **Knowledge Ledger** - Track contributions on-chain
3. **Gradient Validator** - 3 validators approve each FL update
4. **Inference Market** - Understanding as currency
5. **Smart Contracts** - Bounties verified by comprehension

### Philosophy

> Traditional blockchains waste energy on meaningless puzzles.
> Darmiyan validates through **meaning**.

Three nodes don't just vote - they must **comprehend** the same boundary.
The golden ratio φ appears naturally when understanding aligns.

### Links

- **PyPI:** `pip install bazinga`
- **Version:** 4.5.1

---
*Created by Abhi (abhiamu515) | ABHI_AMU = 515 | α⁻¹ = 137*
            """)

if __name__ == "__main__":
    demo.launch()
