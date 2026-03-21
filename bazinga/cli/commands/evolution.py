"""
Evolution command handlers: --propose, --vote, --proposals,
--evolution-status, --constitution.
"""


async def handle_constitution(args):
    """Handle --constitution flag."""
    from ...evolution.constitution import ConstitutionEnforcer
    enforcer = ConstitutionEnforcer()
    print("\n" + "=" * 60)
    print("  BAZINGA CONSTITUTIONAL BOUNDS")
    print("=" * 60)
    print(f"  Total bounds: {len(enforcer.constitution)}")
    print(f"  Forbidden files: {len(enforcer.forbidden_files)}")
    print()
    for b in enforcer.list_bounds():
        print(f"  [{b['name']}]")
        print(f"    {b['description']}")
        print(f"    Patterns: {b['patterns']}")
        print()
    print("  These bounds are IMMUTABLE — no proposal can modify them.")
    print("=" * 60)


async def handle_evolution_status(args):
    """Handle --evolution-status flag."""
    from ...evolution.engine import EvolutionEngine
    engine = EvolutionEngine()
    stats = engine.get_stats()
    auto = stats['autonomy_status']
    print("\n" + "=" * 60)
    print("  BAZINGA EVOLUTION STATUS")
    print("=" * 60)
    print(f"  Autonomy Level: {auto['level_name']} (Level {auto['current_level']})")
    print(f"  Successful Proposals: {auto['successful_proposals']}")
    print(f"  Total Proposals: {auto['total_proposals']}")
    print(f"  Success Rate: {auto['success_rate']:.0%}")
    print(f"  Reverts: {auto['reverts']}")
    print(f"  Node Age: {auto['age_days']:.1f} days")
    if auto.get('next_level_requirements'):
        req = auto['next_level_requirements']
        print(f"\n  Next Level Requirements:")
        print(f"    Proposals: {auto['successful_proposals']}/{req['proposals_needed']}")
        print(f"    Trust: {'?' }/{req['trust_needed']}")
        print(f"    Age: {auto['age_days']:.0f}/{req['days_needed']} days")
    print(f"\n  Proposals by Status: {stats['by_status']}")
    print("=" * 60)


async def handle_proposals(args):
    """Handle --proposals flag."""
    from ...evolution.engine import EvolutionEngine
    engine = EvolutionEngine()
    status_filter = None if args.proposals == 'all' else args.proposals
    proposals = engine.list_proposals(status=status_filter)
    print("\n" + "=" * 60)
    print(f"  BAZINGA PROPOSALS" + (f" (status={args.proposals})" if args.proposals != 'all' else ""))
    print("=" * 60)
    if not proposals:
        print("  No proposals found.")
    else:
        for p in proposals:
            status_icon = {
                'approved': '✓', 'rejected': '✗', 'applied': '◉',
                'voting': '◎', 'reverted': '↺',
            }.get(p.status, '○')
            print(f"\n  {status_icon} {p.proposal_id}")
            print(f"    Title: {p.title}")
            print(f"    Status: {p.status}")
            print(f"    Files: {', '.join(p.modified_files)}")
            if p.ethics_overall is not None:
                print(f"    Ethics: {p.ethics_overall:.2f}")
            print(f"    Votes: {p.approval_count} approve / {p.rejection_count} reject")
    print("\n" + "=" * 60)


async def handle_propose(args):
    """Handle --propose flag."""
    from ...evolution.engine import EvolutionEngine
    from ...evolution.proposal import EvolutionProposal
    import os

    engine = EvolutionEngine()

    file_diffs = []
    if args.diff:
        diff_path = os.path.expanduser(args.diff)
        if os.path.exists(diff_path):
            with open(diff_path) as f:
                content = f.read()
            file_diffs = [{
                "path": diff_path,
                "old_content": "",
                "new_content": content,
            }]
        else:
            print(f"  ✗ Diff file not found: {diff_path}")
            return
    else:
        print("  ✗ --propose requires --diff FILE")
        print("  Usage: bazinga --propose 'Title' --diff path/to/changes.py")
        return

    proposal = EvolutionProposal(
        title=args.propose,
        description=args.propose,
        file_diffs=file_diffs,
        proposer_node_id=getattr(args, 'node_id', None) or 'local',
    )

    result = engine.run_pipeline(proposal)

    print("\n" + "=" * 60)
    print("  PROPOSAL SUBMITTED")
    print("=" * 60)
    print(f"  ID: {result.proposal_id}")
    print(f"  Status: {result.status}")
    print(f"  Constitution: {'PASS' if result.constitution_passes else 'FAIL'}")
    if result.constitution_violations:
        for v in result.constitution_violations:
            print(f"    ✗ {v}")
    if result.ethics_overall is not None:
        print(f"  Ethics: {result.ethics_overall:.2f}")
    if result.sandbox_passed is not None:
        print(f"  Sandbox: {'PASS' if result.sandbox_passed else 'FAIL'}")
    print("=" * 60)


async def handle_vote(args):
    """Handle --vote flag."""
    from ...evolution.engine import EvolutionEngine
    from ...evolution.proposal import Vote

    engine = EvolutionEngine()

    if not args.approve and not args.reject:
        print("  ✗ --vote requires --approve or --reject")
        print("  Usage: bazinga --vote PROP_ID --approve --reason 'Looks good'")
        return

    vote = Vote(
        voter_node_id=getattr(args, 'node_id', None) or 'local',
        approve=args.approve,
        reasoning=args.reason or ("Approved" if args.approve else "Rejected"),
        phi_coherence=0.7,
        trust_weight=0.5,
    )

    accepted = engine.cast_vote(args.vote, vote)
    if accepted:
        print(f"  ✓ Vote cast on {args.vote}: {'APPROVE' if args.approve else 'REJECT'}")
        tally = engine.tally_votes(args.vote)
        print(f"  Tally: {tally.summary}")
    else:
        print(f"  ✗ Could not vote on {args.vote} (not found, already voted, or not in voting)")
