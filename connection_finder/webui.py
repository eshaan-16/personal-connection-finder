"""A tiny, dependency-free local web UI for the connection finder.

Run it with:  python serve.py   (then open http://127.0.0.1:8000)

Endpoints:
  GET  /                     -> the single-page UI
  GET  /estimate?...         -> pre-run cost estimate (drives the slider)
  POST /run  {json}          -> run the pipeline, return ranked connections
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .config import ConfigError, Settings
from .pipeline import find_connectors
from .pricing import MODE_MODELS, plan_run
from .query import build_queries

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Connection Finder</title>
<style>
  :root{
    --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --line:#2a2f3a;
    --text:#e7ebf0; --muted:#98a2b3; --accent:#6ea8fe; --accent2:#7ee0b8;
    --high:#7ee0b8; --med:#f0c674; --low:#8b93a1;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
    font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:860px;margin:0 auto;padding:28px 20px 80px}
  h1{font-size:22px;margin:0 0 4px;letter-spacing:.2px}
  .sub{color:var(--muted);margin:0 0 22px;font-size:13px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
    padding:18px;margin-bottom:16px}
  label{display:block;font-size:12px;color:var(--muted);margin:0 0 6px;
    text-transform:uppercase;letter-spacing:.5px}
  input[type=text]{width:100%;background:var(--panel2);border:1px solid var(--line);
    color:var(--text);border-radius:9px;padding:11px 12px;font-size:15px;outline:none}
  input[type=text]:focus{border-color:var(--accent)}
  .row{display:flex;gap:14px}
  .row>div{flex:1}
  .modes{display:flex;gap:8px;margin-top:2px}
  .mode{flex:1;text-align:center;background:var(--panel2);border:1px solid var(--line);
    border-radius:9px;padding:9px 6px;cursor:pointer;font-size:13px;color:var(--muted)}
  .mode.active{border-color:var(--accent);color:var(--text);background:#1b2432}
  .mode small{display:block;color:var(--muted);font-size:11px;margin-top:2px}
  .sliderrow{display:flex;align-items:center;gap:14px}
  input[type=range]{flex:1;accent-color:var(--accent)}
  .count{font-size:26px;font-weight:600;min-width:46px;text-align:right}
  .cost{margin-top:14px;background:var(--panel2);border:1px solid var(--line);
    border-radius:10px;padding:12px 14px;display:flex;justify-content:space-between;
    align-items:center;flex-wrap:wrap;gap:6px}
  .cost .big{font-size:22px;font-weight:700;color:var(--accent2)}
  .cost .meta{font-size:12px;color:var(--muted)}
  button.run{margin-top:16px;width:100%;background:var(--accent);color:#06101f;
    border:0;border-radius:10px;padding:14px;font-size:16px;font-weight:700;cursor:pointer}
  button.run:disabled{opacity:.6;cursor:default}
  .status{margin:14px 2px;color:var(--muted);font-size:14px;min-height:18px}
  .conn{background:var(--panel);border:1px solid var(--line);border-radius:12px;
    padding:14px 16px;margin-bottom:10px}
  .conn .top{display:flex;align-items:baseline;gap:10px}
  .conn .name{font-size:17px;font-weight:600}
  .pill{font-size:11px;padding:2px 8px;border-radius:20px;border:1px solid var(--line);color:var(--muted)}
  .pill.high{color:var(--high);border-color:#2d5f4c}
  .pill.med{color:var(--med);border-color:#5f5330}
  .pill.low{color:var(--low)}
  .conn .why{color:var(--muted);font-size:13px;margin:6px 0 4px}
  .conn .cat{color:var(--accent);font-size:12px}
  .conn a{color:var(--muted);font-size:12px;text-decoration:none;word-break:break-all}
  .conn a:hover{color:var(--accent)}
  .filtered{color:var(--muted);font-size:13px;margin:6px 2px 18px}
  .spin{display:inline-block;width:14px;height:14px;border:2px solid var(--line);
    border-top-color:var(--accent);border-radius:50%;animation:s .8s linear infinite;vertical-align:-2px;margin-right:8px}
  @keyframes s{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="wrap">
  <h1>Connection Finder</h1>
  <p class="sub">Find a person's reachable niche circle — friends, schoolmates, early-venture and work ties. Public sources only.</p>

  <div class="card">
    <div class="row">
      <div>
        <label>Person</label>
        <input id="person" type="text" placeholder="e.g. Bill Gates" autocomplete="off">
      </div>
      <div>
        <label>Context (disambiguates)</label>
        <input id="context" type="text" placeholder="e.g. Microsoft" autocomplete="off">
      </div>
    </div>

    <div style="margin-top:16px">
      <label>Mode</label>
      <div class="modes" id="modes">
        <div class="mode" data-mode="economy">Economy<small>flash-lite · cheapest</small></div>
        <div class="mode active" data-mode="balanced">Balanced<small>flash · default</small></div>
        <div class="mode" data-mode="accurate">Accurate<small>flash · thorough</small></div>
      </div>
    </div>

    <div style="margin-top:18px">
      <label>Number of connections</label>
      <div class="sliderrow">
        <input id="slider" type="range" min="5" max="40" value="15" step="1">
        <div class="count" id="count">15</div>
      </div>
    </div>

    <div class="cost">
      <div><span class="big" id="costBig">$—</span> <span class="meta" id="costMeta"></span></div>
      <div class="meta" id="costNote"></div>
    </div>

    <button class="run" id="runBtn">Find connections</button>
  </div>

  <div class="status" id="status"></div>
  <div id="results"></div>
</div>

<script>
const $ = id => document.getElementById(id);
let mode = "balanced";

function money(x){ return "$" + (x < 0.01 ? x.toFixed(4) : x.toFixed(3)); }

async function estimate(){
  const n = $("slider").value;
  const person = encodeURIComponent($("person").value);
  const context = encodeURIComponent($("context").value);
  try{
    const r = await fetch(`/estimate?connections=${n}&mode=${mode}&person=${person}&context=${context}`);
    const p = await r.json();
    $("costBig").textContent = money(p.est_total_cost_usd);
    $("costMeta").textContent = `for ~${p.target_connections} connections`;
    const searchTxt = p.est_search_cost_usd > 0 ? money(p.est_search_cost_usd) : "free tier";
    $("costNote").textContent =
      `Gemini ${money(p.est_gemini_cost_usd)} (${p.model}, ~${p.gemini_calls} calls) · `
      + `search ${searchTxt} (${p.n_queries} queries) · ~${p.max_pages_total} pages · re-runs cached ≈ $0`;
  }catch(e){ $("costMeta").textContent = "estimate unavailable"; }
}

$("slider").addEventListener("input", ()=>{ $("count").textContent = $("slider").value; estimate(); });
$("person").addEventListener("input", debounce(estimate, 350));
$("context").addEventListener("input", debounce(estimate, 350));
document.querySelectorAll(".mode").forEach(el=>{
  el.addEventListener("click", ()=>{
    document.querySelectorAll(".mode").forEach(m=>m.classList.remove("active"));
    el.classList.add("active"); mode = el.dataset.mode; estimate();
  });
});
function debounce(fn, ms){ let t; return ()=>{ clearTimeout(t); t=setTimeout(fn, ms); }; }

$("runBtn").addEventListener("click", async ()=>{
  const person = $("person").value.trim(), context = $("context").value.trim();
  if(!person || !context){ $("status").textContent = "Enter a person and a context."; return; }
  $("runBtn").disabled = true;
  $("results").innerHTML = "";
  $("status").innerHTML = `<span class="spin"></span>Searching public sources & extracting connections… (this can take 30–90s)`;
  try{
    const r = await fetch("/run", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({person, context, connections:Number($("slider").value), mode})});
    const d = await r.json();
    if(d.error){ $("status").textContent = "Error: " + d.error; $("runBtn").disabled=false; return; }
    render(d);
  }catch(e){ $("status").textContent = "Request failed: " + e; }
  $("runBtn").disabled = false;
});

function render(d){
  const c = d.cost || {};
  $("status").innerHTML = `<b>${d.connections.length}</b> connections for <b>${d.target}</b> · `
    + `actual cost ${money(c.total_cost_usd||0)} (Gemini ${money(c.gemini_cost_usd||0)}, ${c.gemini_calls||0} calls)`;
  let html = "";
  if(d.removed_famous && d.removed_famous.length){
    html += `<div class="filtered">Filtered ${d.removed_famous.length} well-known people (find them on your own): ${d.removed_famous.slice(0,8).join(", ")}${d.removed_famous.length>8?"…":""}</div>`;
  }
  d.connections.forEach((x,i)=>{
    const tier = (x.tier||"low");
    const src = (x.sources||[]).slice(0,3).map(s=>`<div><a href="${s.url}" target="_blank" rel="noopener">${s.url}</a></div>`).join("");
    html += `<div class="conn">
      <div class="top"><span class="name">${i+1}. ${esc(x.name)}</span>
        <span class="pill ${tier}">${tier.toUpperCase()} ${(x.score||0).toFixed(2)}</span>
        ${x.prominence?`<span class="pill">${esc(x.prominence)}</span>`:""}</div>
      <div class="cat">${esc((x.primary_signal||"").replace(/_/g," "))}</div>
      <div class="why">${esc(x.explanation||"")}</div>
      ${src}
    </div>`;
  });
  if(!d.connections.length) html += `<div class="filtered">No connections found. Try a broader context or raise the count.</div>`;
  $("results").innerHTML = html;
}
function esc(s){ return (s||"").replace(/[&<>"]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
estimate();
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the console quiet
        pass

    # -- helpers --
    def _json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text):
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _n_queries(self, person, context):
        return len(build_queries(person or "Person Name", context or "context"))

    # -- routes --
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            return self._html(PAGE)
        if parsed.path == "/estimate":
            q = parse_qs(parsed.query)
            try:
                connections = int((q.get("connections", ["15"])[0]) or 15)
            except ValueError:
                connections = 15
            connections = max(1, min(80, connections))
            mode = (q.get("mode", ["balanced"])[0]) or "balanced"
            person = (q.get("person", [""])[0])
            context = (q.get("context", [""])[0])
            plan = plan_run(
                connections, mode=mode, n_queries=self._n_queries(person, context),
                max_results_per_query=self.server.base_settings.max_results_per_query,
            )
            return self._json(plan.as_dict())
        return self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/run":
            return self.send_error(404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json({"error": "invalid request body"}, 400)

        person = (data.get("person") or "").strip()
        context = (data.get("context") or "").strip()
        if not person or not context:
            return self._json({"error": "person and context are required"}, 400)
        try:
            connections = max(1, min(80, int(data.get("connections") or 15)))
        except (TypeError, ValueError):
            connections = 15
        mode = data.get("mode") if data.get("mode") in MODE_MODELS else "balanced"

        plan = plan_run(
            connections, mode=mode, n_queries=self._n_queries(person, context),
            max_results_per_query=self.server.base_settings.max_results_per_query,
        )
        base = self.server.base_settings
        settings = Settings.from_env(
            max_pages_total=plan.max_pages_total,
            min_results=plan.min_results,
            gemini_model=plan.model,
            db_path=base.db_path,
        )
        try:
            result = find_connectors(settings, person, context, verbose=False)
        except ConfigError as error:
            return self._json({"error": str(error)}, 400)
        except Exception as error:  # never crash the server on one bad run
            return self._json({"error": f"{type(error).__name__}: {error}"}, 500)

        return self._json({
            "target": result.target,
            "context": result.context,
            "extractor": result.extractor,
            "connections": [c.to_dict() for c in result.scored],
            "removed_famous": [c.candidate.name for c in result.removed_famous],
            "cost": result.cost.as_dict() if result.cost else None,
            "warnings": result.warnings,
        })


def serve(host: str = "127.0.0.1", port: int = 8000, settings: Settings | None = None) -> int:
    """Start the local UI server. Binds to localhost only."""
    settings = settings or Settings.from_env()
    try:
        settings.validate_for_search()
    except ConfigError as error:
        print(f"\n{error}\n", file=sys.stderr)
        return 3
    if not settings.has_gemini():
        print("Note: no GEMINI_API_KEY set — extraction will use the weaker heuristic.", file=sys.stderr)

    server = ThreadingHTTPServer((host, port), _Handler)
    server.base_settings = settings
    url = f"http://{host}:{port}"
    print(f"\n  Connection Finder UI  →  {url}\n  (Ctrl+C to stop)\n", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    finally:
        server.server_close()
    return 0
