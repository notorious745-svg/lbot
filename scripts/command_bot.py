import os, json, re, subprocess, time, pathlib, sys, urllib.request

def sh(*a, **kw): subprocess.run(a, check=True, **kw)
def git(*a): sh("git", *a)

# อ่าน event
with open(os.environ["GH_EVENT_PATH"], "r", encoding="utf-8") as f:
    ev = json.load(f)

repo   = ev["repository"]["full_name"]
base   = ev["repository"]["default_branch"]
actor  = ev["sender"]["login"]
issue  = ev.get("issue") or ev.get("pull_request") or {}
inum   = issue.get("number")
body   = ev.get("comment", {}).get("body", "")
token  = os.environ["GH_TOKEN"]
allow  = [u.strip() for u in os.environ.get("ALLOWED_USERS","").split(",") if u.strip()]
auto_merge = os.environ.get("AUTO_MERGE","true").lower() == "true"

if allow and actor not in allow:
    print(f"actor @{actor} not allowed; skipping.")
    sys.exit(0)

# -------- Parse commands (หลายไฟล์ในคอมเมนต์เดียว) --------
#  /bot write <path> + code block
pat_write = re.compile(r"(?mis)^\s*/bot\s+(write|append)\s+([^\n`]+?)\s*$\s*