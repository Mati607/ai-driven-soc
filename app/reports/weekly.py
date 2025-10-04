from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional

from jinja2 import Template


@dataclass
class TriageResult:
    alert: str
    created_at: Optional[datetime]
    triaged_at: Optional[datetime]
    search_results: List[Dict]
    brief: Optional[str]
    classification: Optional[str]


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def load_triage_results(path: Path) -> List[TriageResult]:
    items: List[TriageResult] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            items.append(
                TriageResult(
                    alert=obj.get("alert", ""),
                    created_at=parse_datetime(obj.get("created_at")),
                    triaged_at=parse_datetime(obj.get("triaged_at")),
                    search_results=obj.get("search_results", []),
                    brief=obj.get("brief"),
                    classification=obj.get("classification"),
                )
            )
    return items


def compute_metrics(items: List[TriageResult], relevance_threshold: float = 0.3) -> Dict:
    mttis: List[float] = []
    relevances: List[float] = []
    for it in items:
        if it.created_at and it.triaged_at:
            mttis.append((it.triaged_at - it.created_at).total_seconds())
        if it.search_results:
            scores = [float(r.get("score", 0.0)) for r in it.search_results]
            if scores:
                relevant = sum(1 for s in scores if s >= relevance_threshold)
                relevances.append(relevant / len(scores))
    return {
        "num_items": len(items),
        "median_mtti": median(mttis) if mttis else None,
        "mean_mtti": mean(mttis) if mttis else None,
        "relevant_match_rate": mean(relevances) if relevances else None,
    }


def render_report(items: List[TriageResult], metrics: Dict) -> str:
    template = Template(
        """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Weekly SOC Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    .kpi { display: flex; gap: 24px; margin-bottom: 24px; }
    .card { padding: 12px 16px; border: 1px solid #ddd; border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border: 1px solid #eee; padding: 8px; text-align: left; }
    th { background: #f8f8f8; }
  </style>
  </head>
<body>
  <h2>Weekly SOC Report</h2>
  <div class="kpi">
    <div class="card"><b>Alerts Triaged</b><div>{{ metrics.num_items or 0 }}</div></div>
    <div class="card"><b>Median MTTI (s)</b><div>{{ '%.0f' % metrics.median_mtti if metrics.median_mtti else 'n/a' }}</div></div>
    <div class="card"><b>Relevant Match Rate</b><div>{{ '%.1f%%' % (100*(metrics.relevant_match_rate or 0)) }}</div></div>
  </div>
  <h3>Recent Alerts</h3>
  <table>
    <thead>
      <tr><th>Alert</th><th>Created</th><th>Triaged</th><th>Top Score</th><th>Classification</th></tr>
    </thead>
    <tbody>
    {% for it in items[:50] %}
      <tr>
        <td>{{ it.alert[:120] }}</td>
        <td>{{ it.created_at }}</td>
        <td>{{ it.triaged_at }}</td>
        <td>{{ '%.3f' % (it.search_results[0].score if it.search_results and it.search_results[0].score is not none else 0) }}</td>
        <td>{{ it.classification or '' }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  <p>Generated on {{ now }}.</p>
</body>
</html>
        """
    )
    # Prepare objects to be template-friendly
    class R:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    pretty_items = []
    for it in items:
        res0 = it.search_results[0] if it.search_results else {"score": 0.0}
        pretty_items.append(
            R({
                "alert": it.alert,
                "created_at": it.created_at,
                "triaged_at": it.triaged_at,
                "search_results": [R({"score": res0.get("score", 0.0)})],
                "classification": it.classification,
            })
        )

    html = template.render(items=pretty_items, metrics=metrics, now=datetime.utcnow())
    return html


def generate_weekly_report(triage_jsonl: Path, output_html: Path) -> None:
    items = load_triage_results(triage_jsonl)
    metrics = compute_metrics(items)
    html = render_report(items, metrics)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")


