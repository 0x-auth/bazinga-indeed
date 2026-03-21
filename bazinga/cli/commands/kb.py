"""
Knowledge Base command handlers: --kb, --kb-sources, --kb-sync, --kb-gmail,
--index, --deindex, --index-public, --scan, --scan-status, --summarize.
"""

from pathlib import Path

from ..utils import _get_kb
from ...constants import PHI, ALPHA
from ...darmiyan import PHI_4


async def handle_kb(args, BAZINGA_cls):
    """Handle --kb and related flags."""
    BazingaKB = _get_kb()
    kb = BazingaKB()

    if args.kb_sources:
        kb.show_sources()
        return

    if args.kb_sync:
        kb.sync_all()
        return

    # Handle --kb-gmail with optional direct query
    if args.kb_gmail is not None:
        sources = ['gmail']
        query = args.kb_gmail if args.kb_gmail else args.kb
        if query:
            results = kb.search(query, sources=sources)
            kb.display_results(results, query)
        else:
            kb.show_sources()
            print("\nUsage: bazinga --kb-gmail \"your query here\"")
        return

    if args.kb is not None:
        sources = []
        if args.kb_gdrive:
            sources.append('gdrive')
        if args.kb_mac:
            sources.append('mac')
        # --kb-phone as filter
        kb_phone_is_path = hasattr(args, 'kb_phone') and args.kb_phone and args.kb_phone != '' and '/' in args.kb_phone
        if hasattr(args, 'kb_phone') and args.kb_phone is not None and not kb_phone_is_path:
            sources.append('phone')

        if not sources:
            sources = None

        if args.kb == '':
            kb.show_sources()
            print("\nUsage: bazinga --kb \"your query here\"")
        else:
            results = kb.search(args.kb, sources=sources)

            if hasattr(args, 'summarize') and args.summarize and results:
                print(f"\n🔍 Searching KB for: \"{args.kb}\"...")
                context_parts = []
                for r in results[:10]:
                    content = r.get('content', r.get('text', ''))[:500]
                    source = r.get('file', r.get('source', 'unknown'))
                    if content:
                        context_parts.append(f"[From {source}]: {content}")

                if context_parts:
                    context = "\n\n".join(context_parts)
                    prompt = f"""Based on the following information from the user's personal knowledge base, answer this question: "{args.kb}"

Context from KB:
{context}

Provide a concise, helpful answer based on the above context. If the context doesn't contain relevant information, say so."""

                    bazinga = BAZINGA_cls(verbose=args.verbose)
                    answer = await bazinga.ask(prompt, fresh=True)
                    print(f"\n📚 Based on your data:\n")
                    print(f"{answer}\n")
                    print(f"  (Found {len(results)} relevant documents)")
                else:
                    print(f"\n  No content found for '{args.kb}'")
            else:
                kb.display_results(results, args.kb)


async def handle_kb_phone_path(args):
    """Handle --kb-phone-path flag."""
    import os
    BazingaKB = _get_kb()
    kb = BazingaKB()
    kb.set_phone_data_path(os.path.expanduser(args.kb_phone_path))
    kb.show_sources()


async def handle_kb_phone_as_path(args):
    """Handle --kb-phone when it contains a path."""
    import os
    BazingaKB = _get_kb()
    kb = BazingaKB()
    kb.set_phone_data_path(os.path.expanduser(args.kb_phone))
    kb.show_sources()


async def handle_index(args, BAZINGA_cls):
    """Handle --index flag."""
    bazinga = BAZINGA_cls(verbose=args.verbose)
    await bazinga.index(args.index)


async def handle_deindex(args, BAZINGA_cls):
    """Handle --deindex flag."""
    from pathlib import Path as _Path
    bazinga = BAZINGA_cls(verbose=args.verbose)
    collection = bazinga.ai.collection
    if not collection:
        print("No vector database found.")
        return

    total_removed = 0
    for path_str in args.deindex:
        prefix = str(_Path(path_str).expanduser().resolve())
        all_data = collection.get(include=['metadatas'])
        ids_to_remove = []
        for doc_id, meta in zip(all_data['ids'], all_data['metadatas']):
            source = meta.get('source_file', '')
            if source.startswith(prefix):
                ids_to_remove.append(doc_id)

        if ids_to_remove:
            for i in range(0, len(ids_to_remove), 5000):
                batch = ids_to_remove[i:i+5000]
                collection.delete(ids=batch)
            total_removed += len(ids_to_remove)
            print(f"  Removed {len(ids_to_remove)} chunks from: {prefix}")
        else:
            print(f"  No indexed chunks found from: {prefix}")

    print(f"\n  Total removed: {total_removed} chunks")
    remaining = collection.count()
    print(f"  Remaining in index: {remaining} chunks")


async def handle_index_public(args):
    """Handle --index-public flag."""
    from ...public_knowledge import index_public_knowledge, get_preset_topics, TOPIC_PRESETS, ARXIV_PRESETS

    source = args.index_public
    presets = ARXIV_PRESETS if source == "arxiv" else TOPIC_PRESETS

    if args.topics:
        if args.topics.lower() in presets:
            topics = get_preset_topics(args.topics, source)
            print(f"Using preset '{args.topics}': {', '.join(topics)}")
        else:
            topics = [t.strip() for t in args.topics.split(",")]
    else:
        topics = get_preset_topics("bazinga", source)
        print(f"Using default BAZINGA topics: {', '.join(topics)}")

    result = await index_public_knowledge(source, topics, verbose=True)

    if result.get("error"):
        print(f"\nError: {result['error']}")
    else:
        count_key = "total_articles" if source == "wikipedia" else "total_papers"
        count = result.get(count_key, result.get("total_articles", result.get("total_papers", 0)))
        item_type = "papers" if source == "arxiv" else "articles"
        print(f"\n✅ Indexed {count} {item_type} ({result.get('total_chunks', 0)} chunks)")
        print(f"   Now run 'bazinga --publish' to share with the network!")


async def handle_scan(args):
    """Handle --scan flag."""
    from ...knowledge import get_scanner
    scanner = get_scanner()
    print()
    print("◊ KB DNA Scanner — Semantic Compression")
    print(f"  φ = {PHI} | α = {ALPHA}")
    print()
    scanner.scan(args.scan, max_depth=args.scan_depth, verbose=True)
    print("◊ Manifests ready! Your ask queries now have breadth context.")
    print()


async def handle_scan_status(args):
    """Handle --scan-status flag."""
    from ...knowledge import get_scanner
    scanner = get_scanner()
    status = scanner.get_status()
    print()
    if not status['exists']:
        print("  No KB manifest found.")
        print("  Run: bazinga --scan ~/Documents ~/Projects")
    else:
        print("◊ KB Manifest Status")
        print(f"  Generated: {status['generated']}")
        print(f"  Files: {status['total_files']}")
        print(f"  Words: {status['total_words']:,}")
        print(f"  Projects: {status['projects']}")
        print(f"  Manifest size: {status['manifest_size_mb']:.2f} MB")
        print()
        if status.get('genes'):
            print("  Gene distribution:")
            for gene, count in sorted(status['genes'].items(), key=lambda x: -x[1]):
                print(f"    {gene}: {count}")
        print()
        if status.get('scanned_paths'):
            print("  Scanned paths:")
            for p in status['scanned_paths']:
                print(f"    {p}")
    print()
