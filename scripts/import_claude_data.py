#!/usr/bin/env python3
"""
Import Claude conversation data into BAZINGA KB.

Handles the Claude export format:
[
  {
    "uuid": "...",
    "name": "conversation title",
    "chat_messages": [
      {"text": "...", "sender": "human/assistant", ...}
    ]
  }
]
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def extract_conversations(json_file: Path, output_dir: Path, verbose: bool = True):
    """Extract conversations from Claude JSON export to individual text files."""

    if verbose:
        print(f"\n◊ Processing: {json_file.name}")
        print(f"  Size: {json_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Stream-parse the JSON to handle large files
    conversations = []
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Try to fix common JSON issues
            if content.strip().endswith(',]'):
                content = content.strip()[:-2] + ']'
            conversations = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON parse error at position {e.pos}: {e.msg}")
        # Try to parse up to the error
        try:
            partial = content[:e.pos-100]
            # Find last complete object
            last_bracket = partial.rfind('}')
            if last_bracket > 0:
                partial = partial[:last_bracket+1] + ']'
                conversations = json.loads(partial)
                print(f"  → Recovered {len(conversations)} conversations before error")
        except:
            print(f"  ✗ Could not recover partial data")
            return 0

    if verbose:
        print(f"  Found: {len(conversations)} conversations")

    extracted = 0
    for conv in conversations:
        if not isinstance(conv, dict):
            continue

        conv_id = conv.get('uuid', '')[:8]
        conv_name = conv.get('name', 'untitled')
        messages = conv.get('chat_messages', [])

        if not messages:
            continue

        # Clean filename
        safe_name = "".join(c if c.isalnum() or c in ' -_' else '_' for c in conv_name)[:50]
        filename = f"{conv_id}_{safe_name}.txt"

        # Build conversation text
        lines = []
        lines.append(f"# Conversation: {conv_name}")
        lines.append(f"# ID: {conv.get('uuid', 'unknown')}")
        lines.append(f"# Created: {conv.get('created_at', 'unknown')}")
        lines.append("")

        for msg in messages:
            sender = msg.get('sender', 'unknown').upper()
            text = msg.get('text', '')
            if text:
                lines.append(f"[{sender}]")
                lines.append(text)
                lines.append("")

        # Write file
        output_file = output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        extracted += 1

    if verbose:
        print(f"  Extracted: {extracted} conversations to text files")

    return extracted


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Import Claude data to BAZINGA KB')
    parser.add_argument('input_dir', help='Directory with Claude JSON exports')
    parser.add_argument('--output', '-o', default=None, help='Output directory for extracted text')
    parser.add_argument('--index', action='store_true', help='Also index into BAZINGA KB')
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.exists():
        print(f"Error: {input_dir} not found")
        sys.exit(1)

    # Default output to ~/.bazinga/claude_data/
    output_dir = Path(args.output) if args.output else Path.home() / '.bazinga' / 'claude_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  BAZINGA Claude Data Importer")
    print("=" * 60)
    print(f"\n  Input:  {input_dir}")
    print(f"  Output: {output_dir}")

    total_extracted = 0

    # Process all JSON files
    json_files = list(input_dir.glob('*.json'))
    print(f"\n  Found {len(json_files)} JSON files\n")

    for json_file in sorted(json_files):
        if json_file.name.startswith('.'):
            continue
        extracted = extract_conversations(json_file, output_dir, verbose=True)
        total_extracted += extracted

    print("\n" + "=" * 60)
    print(f"  Total conversations extracted: {total_extracted}")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)

    # Optionally index
    if args.index:
        print("\n◊ Indexing into BAZINGA KB...")
        os.system(f'python3.13 -m bazinga --index "{output_dir}"')
    else:
        print(f"\n  To index: bazinga --index \"{output_dir}\"")


if __name__ == '__main__':
    main()
