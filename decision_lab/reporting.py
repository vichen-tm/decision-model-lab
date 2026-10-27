"""Small Markdown/CSV/SVG report for saved responses."""
import csv, html, json
from pathlib import Path
from .analysis import analyze

def write_report(rows, directory):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    result = analyze(rows)
    (out / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    fields = ['model', 'n', 'accepted', 'correct', 'errors', 'accuracy', 'cost_usd', 'unknown_cost_calls']
    with (out / 'models.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m, r in result['models'].items():
            w.writerow({'model': m, **{k: r[k] for k in fields[1:]}})
    curves = []
    for m, r in result['models'].items():
        for x in r['confidence_curve']:
            curves.append({'model': m, **x})
    if curves:
        with (out / 'confidence.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(curves[0]))
            w.writeheader()
            w.writerows(curves)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 460"><rect width="800" height="460" fill="white"/><g font-family="sans-serif" font-size="14" fill="#222"><text x="65" y="24">Accuracy among accepted vs automatic coverage (descriptive)</text><path d="M65 50V370H740" fill="none" stroke="#555"/>']
    for tick in range(0, 101, 20):
        x = 65 + 675 * tick / 100
        y = 370 - 320 * tick / 100
        svg.extend([f'<text x="{x}" y="390">{tick}%</text>', f'<text x="18" y="{y}">{tick}%</text>'])
    colors = ['#1757a6', '#af301e', '#087b55', '#873ba1', '#a46a00']
    for i, (m, r) in enumerate(result['models'].items()):
        color = colors[i % len(colors)]
        pts = ' '.join((f"{65 + 675 * z['coverage']:.2f},{370 - 320 * z['accuracy']:.2f}" for z in r['confidence_curve'] if z['accuracy'] is not None))
        if pts:
            svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<text x="{65 + i % 3 * 230}" y="{420 + i // 3 * 18}" fill="{color}">{html.escape(m)}</text>')
    svg.append('</g></svg>')
    (out / 'confidence.svg').write_text(''.join(svg))
    lines = ['# Saved-response report', '', 'These curves are descriptive. Choose thresholds on development data and validate frozen rules on independent data.', '', '| Model | Correct / total | Accuracy | Recorded cost USD |', '|---|---:|---:|---:|']
    for m, r in result['models'].items():
        lines.append(f"| {m.replace('|', '/')} | {r['correct']} / {r['n']} | {r['accuracy']:.2%} | {(r['cost_usd'] if r['cost_usd'] is not None else 'unknown')} |")
    lines += ['', '![Confidence curve](confidence.svg)', '', 'CSV files contain exact values; `analysis.json` includes predicted-class metrics and error overlap.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    return {'output': str(out), 'models': len(result['models'])}
