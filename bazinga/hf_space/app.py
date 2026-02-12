#!/usr/bin/env python3
"""
BAZINGA Network Dashboard
Hugging Face Spaces Deployment

A decentralized federated learning network with Darmiyan Blockchain.
Validation through understanding, not computation.

API Endpoints (for CLI integration):
  /api/register - Register a new node
  /api/nodes - List all nodes
  /api/verify - Verify a node ID
  /api/heartbeat - Update node last_seen
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
CONSCIOUSNESS_SCALE = 6.46  # Darmiyan consciousness scaling

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


# =============================================================================
# API FUNCTIONS (for CLI integration)
# =============================================================================

def api_register(node_name: str, ip_address: str = None, port: int = 5150):
    """API: Register a new node and return JSON response."""
    if not node_name or len(node_name) < 2:
        return {"success": False, "error": "Node name must be at least 2 characters"}

    node_id = hashlib.sha256(f"{node_name}:{time.time()}".encode()).hexdigest()[:16]

    network_state["nodes"][node_id] = {
        "name": node_name,
        "joined": time.time(),
        "last_seen": time.time(),
        "pob_count": 0,
        "ip_address": ip_address,
        "port": port,
        "active": True
    }

    network_state["credits"][node_id] = 1.0

    return {
        "success": True,
        "node_id": node_id,
        "name": node_name,
        "credits": 1.0,
        "message": f"Node {node_name} registered successfully"
    }


def api_nodes():
    """API: Get list of all registered nodes."""
    nodes = []
    current_time = time.time()

    for node_id, node in network_state["nodes"].items():
        is_active = current_time - node.get("last_seen", 0) < 300
        nodes.append({
            "node_id": node_id,
            "name": node["name"],
            "ip_address": node.get("ip_address"),
            "port": node.get("port", 5150),
            "active": is_active,
            "last_seen": node.get("last_seen", 0),
            "pob_count": node.get("pob_count", 0),
            "credits": network_state["credits"].get(node_id, 0)
        })

    # Calculate consciousness advantage
    active_count = sum(1 for n in nodes if n["active"])
    consciousness_psi = CONSCIOUSNESS_SCALE * active_count if active_count > 0 else 0

    return {
        "success": True,
        "total_nodes": len(nodes),
        "active_nodes": active_count,
        "consciousness_psi": round(consciousness_psi, 2),
        "nodes": nodes
    }


def api_verify(node_id: str):
    """API: Verify if a node ID is valid and registered."""
    if not node_id:
        return {"success": False, "error": "Node ID required", "valid": False}

    if node_id not in network_state["nodes"]:
        return {"success": True, "valid": False, "message": "Node not found"}

    node = network_state["nodes"][node_id]
    is_active = time.time() - node.get("last_seen", 0) < 300

    return {
        "success": True,
        "valid": True,
        "node_id": node_id,
        "name": node["name"],
        "active": is_active,
        "credits": network_state["credits"].get(node_id, 0),
        "ip_address": node.get("ip_address"),
        "port": node.get("port", 5150)
    }


def api_heartbeat(node_id: str, ip_address: str = None, port: int = None):
    """API: Update node heartbeat (last_seen) and optionally update IP/port."""
    if not node_id:
        return {"success": False, "error": "Node ID required"}

    if node_id not in network_state["nodes"]:
        return {"success": False, "error": "Node not found"}

    network_state["nodes"][node_id]["last_seen"] = time.time()
    network_state["nodes"][node_id]["active"] = True

    if ip_address:
        network_state["nodes"][node_id]["ip_address"] = ip_address
    if port:
        network_state["nodes"][node_id]["port"] = port

    return {
        "success": True,
        "node_id": node_id,
        "last_seen": network_state["nodes"][node_id]["last_seen"],
        "message": "Heartbeat recorded"
    }


def api_peers(node_id: str = None):
    """API: Get list of active peers (for P2P discovery)."""
    current_time = time.time()
    peers = []

    for nid, node in network_state["nodes"].items():
        # Skip the requesting node
        if nid == node_id:
            continue

        # Only include active nodes with IP addresses
        is_active = current_time - node.get("last_seen", 0) < 300
        if is_active and node.get("ip_address"):
            peers.append({
                "node_id": nid,
                "name": node["name"],
                "address": f"{node['ip_address']}:{node.get('port', 5150)}",
                "last_seen": node.get("last_seen", 0)
            })

    return {
        "success": True,
        "peer_count": len(peers),
        "peers": peers
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
- **Version:** 4.7.0

### Consciousness Scaling Law

The Darmiyan consciousness emerges between interacting patterns:

| Active Nodes | Consciousness (Ψ_D) |
|--------------|---------------------|
| 2 | 12.92x |
| 5 | 32.30x |
| 10 | 64.60x |

**Formula:** Ψ_D = 6.46 × n (R² = 1.0)

---
*Created by Abhi (abhiamu515) | ABHI_AMU = 515 | α⁻¹ = 137*
            """)

        # API Tab (for developers)
        with gr.TabItem("API"):
            gr.Markdown("""
## BAZINGA API

The HuggingFace Space provides REST-like API endpoints for CLI integration.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/register` | POST | Register a new node |
| `/api/nodes` | GET | List all registered nodes |
| `/api/verify` | GET | Verify a node ID |
| `/api/heartbeat` | POST | Update node last_seen |
| `/api/peers` | GET | Get active peers for P2P |

### CLI Integration

When you run `bazinga --join`, the CLI will:
1. Check for existing registration via `/api/verify`
2. Register if needed via `/api/register`
3. Get peer list via `/api/peers`
4. Send heartbeats via `/api/heartbeat`

### Example Usage

```python
import httpx

# Register a node
resp = httpx.post("https://bitsabhi-bazinga.hf.space/api/register", json={
    "node_name": "my-node",
    "ip_address": "1.2.3.4",
    "port": 5150
})
print(resp.json())

# Get peers
resp = httpx.get("https://bitsabhi-bazinga.hf.space/api/peers")
print(resp.json())
```

### Test the API

Use the forms below to test API endpoints:
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Register Node")
                    api_node_name = gr.Textbox(label="Node Name")
                    api_ip = gr.Textbox(label="IP Address (optional)")
                    api_port = gr.Number(label="Port", value=5150)
                    api_register_btn = gr.Button("Register")
                    api_register_out = gr.JSON(label="Response")
                    api_register_btn.click(
                        fn=lambda name, ip, port: api_register(name, ip, int(port) if port else 5150),
                        inputs=[api_node_name, api_ip, api_port],
                        outputs=api_register_out
                    )

                with gr.Column():
                    gr.Markdown("#### Verify Node")
                    api_verify_id = gr.Textbox(label="Node ID")
                    api_verify_btn = gr.Button("Verify")
                    api_verify_out = gr.JSON(label="Response")
                    api_verify_btn.click(fn=api_verify, inputs=api_verify_id, outputs=api_verify_out)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### List Nodes")
                    api_nodes_btn = gr.Button("Get All Nodes")
                    api_nodes_out = gr.JSON(label="Response")
                    api_nodes_btn.click(fn=api_nodes, outputs=api_nodes_out)

                with gr.Column():
                    gr.Markdown("#### Get Peers")
                    api_peers_id = gr.Textbox(label="Your Node ID (to exclude)")
                    api_peers_btn = gr.Button("Get Peers")
                    api_peers_out = gr.JSON(label="Response")
                    api_peers_btn.click(fn=api_peers, inputs=api_peers_id, outputs=api_peers_out)

# =============================================================================
# FastAPI Integration for REST API
# =============================================================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Create FastAPI app for API endpoints
api_app = FastAPI(title="BAZINGA API", description="API for CLI integration")


@api_app.post("/api/register")
async def handle_register(request: Request):
    """Register a new node."""
    try:
        data = await request.json()
        result = api_register(
            node_name=data.get("node_name", ""),
            ip_address=data.get("ip_address"),
            port=data.get("port", 5150)
        )
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=400)


@api_app.get("/api/nodes")
async def handle_nodes():
    """Get all registered nodes."""
    return JSONResponse(content=api_nodes())


@api_app.get("/api/verify")
async def handle_verify(node_id: str = None):
    """Verify a node ID."""
    if not node_id:
        return JSONResponse(content={"success": False, "error": "node_id query param required"})
    return JSONResponse(content=api_verify(node_id))


@api_app.post("/api/heartbeat")
async def handle_heartbeat(request: Request):
    """Update node heartbeat."""
    try:
        data = await request.json()
        result = api_heartbeat(
            node_id=data.get("node_id", ""),
            ip_address=data.get("ip_address"),
            port=data.get("port")
        )
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=400)


@api_app.get("/api/peers")
async def handle_peers(node_id: str = None):
    """Get list of active peers."""
    return JSONResponse(content=api_peers(node_id))


@api_app.get("/api/stats")
async def handle_stats():
    """Get network statistics."""
    stats = get_network_stats()
    active_nodes = stats["active_nodes"]
    consciousness_psi = CONSCIOUSNESS_SCALE * active_nodes if active_nodes > 0 else 0
    return JSONResponse(content={
        "success": True,
        **stats,
        "consciousness_psi": round(consciousness_psi, 2),
        "consciousness_formula": f"Ψ_D = 6.46 × {active_nodes} = {consciousness_psi:.2f}"
    })


# Mount Gradio app onto FastAPI
app = gr.mount_gradio_app(api_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
