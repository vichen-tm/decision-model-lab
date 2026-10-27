"""Command-line entry point; only `run --send` constructs a live transport."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from .analysis import read_rows, analyze, apply, metrics, compare

def dump(value, path=None):
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    else:
        print(text, end='')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest='command', required=True)
    a = s.add_parser('analyze')
    a.add_argument('responses')
    a.add_argument('--output')
    a = s.add_parser('report')
    a.add_argument('responses')
    a.add_argument('--output', required=True)
    a = s.add_parser('compare')
    a.add_argument('before')
    a.add_argument('after')
    a.add_argument('--output')
    a = s.add_parser('evaluate')
    a.add_argument('responses')
    a.add_argument('policy')
    a.add_argument('--referral-cost', type=float)
    a.add_argument('--output')
    a = s.add_parser('split')
    a.add_argument('dataset')
    a.add_argument('--output', required=True)
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--final-fraction', type=float, default=0.25)
    a = s.add_parser('prepare')
    a.add_argument('dataset')
    a.add_argument('routes')
    a.add_argument('--run-dir', required=True)
    a.add_argument('--concurrency', type=int, default=1)
    a = s.add_parser('run')
    a.add_argument('--run-dir', required=True)
    a.add_argument('--send', action='store_true')
    a.add_argument('--retry-technical', action='store_true')
    a = s.add_parser('report-run')
    a.add_argument('--run-dir', required=True)
    a = s.add_parser('export-run')
    a.add_argument('--run-dir', required=True)
    a.add_argument('--output', required=True)
    a = s.add_parser('resolve')
    a.add_argument('--run-dir', required=True)
    a.add_argument('--task-id', required=True)
    a.add_argument('--state', choices=['unknown', 'failed'], required=True)
    a.add_argument('--action', choices=['retry', 'abandon'], required=True)
    a = s.add_parser('estimate')
    a.add_argument('--requests', type=int, required=True)
    a.add_argument('--input-tokens', type=int, required=True)
    a.add_argument('--output-tokens', type=int, required=True)
    a.add_argument('--input-per-million', type=float, required=True)
    a.add_argument('--output-per-million', type=float, required=True)
    args = p.parse_args()
    try:
        if args.command == 'analyze':
            dump(analyze(read_rows(args.responses)), args.output)
        elif args.command == 'report':
            from .reporting import write_report
            dump(write_report(read_rows(args.responses), args.output))
        elif args.command == 'compare':
            dump(compare(read_rows(args.before), read_rows(args.after)), args.output)
        elif args.command == 'evaluate':
            rows = read_rows(args.responses)
            policy = json.loads(Path(args.policy).read_text())
            d, c = apply(rows, policy)
            if args.referral_cost is not None and (not __import__('math').isfinite(args.referral_cost) or args.referral_cost < 0):
                raise ValueError('Invalid referral price')
            dump({'policy': policy, 'metrics': metrics(rows, d, c, args.referral_cost), 'decisions': [{'id': r['id'], 'prediction': v, 'calls': u} for r, v, u in zip(rows, d, c)]}, args.output)
        elif args.command == 'estimate':
            vals = [args.requests, args.input_tokens, args.output_tokens, args.input_per_million, args.output_per_million]
            if any((not __import__('math').isfinite(x) or x < 0 for x in vals)):
                raise ValueError('Negative or nonfinite estimate input')
            dump({'requests': args.requests, 'estimated_usd': args.requests * (args.input_tokens * args.input_per_million + args.output_tokens * args.output_per_million) / 1000000.0, 'basis': 'User-supplied prices; excludes retries and referral'})
        elif args.command == 'split':
            from decision_benchmark.runner.dataset import load_dataset
            rows = list(load_dataset(Path(args.dataset)).values())
            if not 0 < args.final_fraction < 1:
                raise ValueError('final fraction must be between zero and one')
            groups = {}
            for r in rows:
                group = r.get('group_id')
                if group is None:
                    group = r['id']
                if not isinstance(group, str) or not group.strip():
                    raise ValueError('group_id must be a nonempty string or null')
                groups.setdefault(group, []).append(r)
            keys = sorted(groups)
            random.Random(args.seed).shuffle(keys)
            if len(keys) < 2:
                raise ValueError('Need two independent groups')
            cut = max(1, min(len(keys) - 1, round(len(keys) * args.final_fraction)))
            final = set(keys[:cut])
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=False)
            for name, selected in [('final', final), ('working', set(keys) - final)]:
                (out / (name + '.jsonl')).write_text(''.join((json.dumps(r) + '\n' for k in sorted(selected) for r in groups[k])))
            dump({'seed': args.seed, 'group_key': 'group_id or id', 'files': {n: hashlib.sha256((out / (n + '.jsonl')).read_bytes()).hexdigest() for n in ['working', 'final']}}, out / 'manifest.json')
        elif args.command == 'prepare':
            from decision_benchmark.runner.dataset import load_dataset
            from decision_benchmark.runner.execution import prepare
            cases = load_dataset(Path(args.dataset))
            routes = json.loads(Path(args.routes).read_text())
            dump(prepare(Path('.'), Path(args.run_dir), [r['model'] for r in routes], sorted({k[0] for k in cases}), None, 1, args.concurrency, routes=routes, cases=cases))
        elif args.command == 'run':
            if not args.send:
                raise ValueError('No requests sent: --send required')
            from decision_benchmark.runner.execution import run
            from decision_benchmark.runner.transport import OpenRouterTransport
            result = run(Path(args.run_dir), OpenRouterTransport(os.environ.get('OPENROUTER_API_KEY')), args.retry_technical)
            dump(result)
            if result.get('status') != 'complete':
                raise SystemExit(2)
        elif args.command == 'report-run':
            from decision_benchmark.runner.report import report_run
            dump(report_run(Path(args.run_dir)))
        elif args.command == 'resolve':
            from decision_benchmark.runner.execution import resolve_attempt
            dump(resolve_attempt(Path(args.run_dir), args.task_id, args.action, args.state))
        elif args.command == 'export-run':
            from .export import export_run
            rows = export_run(Path(args.run_dir))
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(''.join((json.dumps(r) + '\n' for r in rows)))
            dump({'rows': len(rows), 'output': args.output})
    except (ValueError, KeyError, RuntimeError) as e:
        p.exit(1, str(e) + '\n')
if __name__ == '__main__':
    main()
