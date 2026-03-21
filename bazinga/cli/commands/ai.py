"""
AI command handlers: --ask, --chat, --multi-ai, --code, --quantum,
--coherence, --vac, --generate, --demo, interactive mode.
"""

from pathlib import Path

from ..utils import _get_kb
from ...constants import VAC_SEQUENCE
from ...darmiyan import PHI_4


async def handle_quantum(args, BAZINGA_cls):
    """Handle --quantum flag."""
    bazinga = BAZINGA_cls(verbose=args.verbose)

    quantum_input = args.quantum
    kb_context = ""

    if args.kb and args.kb != '':
        kb = _get_kb()()
        kb_results = kb.search(args.kb, limit=5)
        if kb_results:
            kb_texts = [f"{r.get('title', '')}: {r.get('content', '')[:200]}" for r in kb_results[:3]]
            kb_context = " | ".join(kb_texts)
            quantum_input = f"{args.quantum} [KB Context: {kb_context[:500]}]"
            print(f"\n📚 KB Search: \"{args.kb}\" → {len(kb_results)} results piped to quantum analyzer")

    result = bazinga.quantum_analyze(quantum_input)
    print(f"\nQuantum Analysis:")
    print(f"  Input: {result['input'][:100]}{'...' if len(result['input']) > 100 else ''}")
    print(f"  Essence: {result['essence']}")
    print(f"  Probability: {result['probability']:.2%}")
    print(f"  Coherence: {result['coherence']:.4f}")
    print(f"  Entangled: {', '.join(result['entangled'][:5])}")

    if result['coherence'] > 0.5:
        try:
            from ...blockchain import create_chain, create_wallet
            from ...blockchain.miner import auto_attest_if_coherent
            chain = create_chain()
            wallet = create_wallet()
            attested = auto_attest_if_coherent(
                chain=chain,
                content=quantum_input,
                summary=f"Quantum essence: {result['essence']}",
                sender=wallet.node_id,
                coherence=result['coherence'],
            )
            if attested:
                print(f"\n  ⛓️  Auto-attested to chain (coherence {result['coherence']:.3f} > 0.5)")
        except Exception:
            pass


async def handle_coherence(args, BAZINGA_cls):
    """Handle --coherence flag."""
    bazinga = BAZINGA_cls(verbose=args.verbose)
    result = bazinga.check_coherence(args.coherence)
    print(f"\nΛG Coherence Check:")
    print(f"  Input: {result['input']}")
    print(f"  Total Coherence: {result['total_coherence']:.3f}")
    print(f"  Entropic Deficit: {result['entropic_deficit']:.3f}")
    print(f"  V.A.C. Achieved: {result['is_vac']}")
    print(f"  Boundaries:")
    for name, value in result['boundaries'].items():
        print(f"    {name}: {value:.3f}")


async def handle_code(args):
    """Handle --code flag."""
    try:
        from ...intelligent_coder import IntelligentCoder
        coder = IntelligentCoder()
        lang = {'js': 'javascript', 'ts': 'typescript'}.get(args.lang, args.lang)
        print(f"Generating {lang} code...")
        result = await coder.generate(args.code, lang)
        print(f"\n# Provider: {result.provider}")
        print(f"# Coherence: {result.coherence:.3f}")
        print()
        print(result.code)
    except ImportError as e:
        print(f"Error: Intelligent coder not available: {e}")


async def handle_generate(args):
    """Handle --generate flag."""
    from ...tui import CodeGenerator
    gen = CodeGenerator()
    lang = 'javascript' if args.lang == 'js' else args.lang
    code = gen.generate(args.generate, lang)
    print(code)


async def handle_vac(args, BAZINGA_cls):
    """Handle --vac flag."""
    bazinga = BAZINGA_cls(verbose=args.verbose)
    print(f"Testing V.A.C.: {VAC_SEQUENCE}")
    result = bazinga.check_coherence(VAC_SEQUENCE)
    print(f"  Coherence: {result['total_coherence']:.3f}")
    print(f"  V.A.C. Achieved: {result['is_vac']}")


async def handle_multi_ai(args):
    """Handle --multi-ai flag."""
    print(f"\n🤖 BAZINGA INTER-AI CONSENSUS")
    print(f"=" * 60)
    print(f"  Multiple AIs reaching understanding through φ-coherence")
    print()

    try:
        from ...inter_ai import InterAIConsensus

        query = args.multi_ai
        if args.file:
            file_path = Path(args.file).expanduser()
            if file_path.exists():
                file_content = file_path.read_text()[:8000]
                query = f"{args.multi_ai}\n\n[File: {args.file}]\n```\n{file_content}\n```"
                print(f"  📄 File context: {args.file}")
            else:
                print(f"  ⚠ File not found: {args.file}")

        consensus = InterAIConsensus(verbose=True)
        result = await consensus.ask(query)
        consensus.export_log("bazinga_consensus.json")

    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Make sure httpx is installed: pip install httpx")


async def handle_chat(args, BAZINGA_cls, _start_background_p2p, _start_query_server,
                      _stop_background_p2p, _mesh_query_getter):
    """Handle --chat flag."""
    _start_background_p2p()

    bazinga = BAZINGA_cls(verbose=args.verbose)
    if args.local:
        bazinga.use_local = True

    await _start_query_server(bazinga)

    try:
        if not args.simple:
            try:
                from ...tui import run_tui_async
                await run_tui_async(
                    bazinga_instance=bazinga,
                    mode="chat",
                    mesh_query=_mesh_query_getter(),
                )
                return
            except ImportError:
                pass

        await bazinga.chat_interactive()
    finally:
        _stop_background_p2p()


async def handle_ask(args, BAZINGA_cls):
    """Handle --ask or positional question."""
    bazinga = BAZINGA_cls(verbose=args.verbose)
    if args.local:
        bazinga.use_local = True

    question = args.ask or args.question
    query = question
    if args.file:
        file_path = Path(args.file).expanduser()
        if file_path.exists():
            file_content = file_path.read_text()[:8000]
            query = f"{question}\n\n[File: {args.file}]\n```\n{file_content}\n```"
            print(f"  📄 File context: {args.file}")
        else:
            print(f"  ⚠ File not found: {args.file}")

    response = await bazinga.ask(query, fresh=args.fresh)
    print(f"\n{response}\n")


async def handle_demo(args, BAZINGA_cls):
    """Handle --demo flag."""
    bazinga = BAZINGA_cls(verbose=args.verbose)
    print("Running demo...")
    await bazinga.index([str(Path(__file__).parent.parent)])
    response = await bazinga.ask("What is BAZINGA?")
    print(f"\n{response}\n")


async def handle_interactive(args, BAZINGA_cls):
    """Handle default interactive mode."""
    bazinga = BAZINGA_cls(verbose=args.verbose)
    if args.local:
        bazinga.use_local = True

    if args.simple:
        await bazinga.interactive()
    else:
        try:
            from ...tui import run_tui_async
            await run_tui_async(bazinga_instance=bazinga, mode="interactive")
        except ImportError:
            await bazinga.interactive()
