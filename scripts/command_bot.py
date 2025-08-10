import os, json, re, subprocess, time, pathlib, sys, urllib.request

def sh(*a):
    subprocess.run(a, check=True)

# อ่านอีเวนต์จาก GitHub
with open(os.environ["GH_EVENT_PATH"], "r", encoding="utf-8") as f:
    ev = json.load(f)

actor = ev["sender"]["login"]
allow = [u.strip() for u in os.environ.get("ALLOWED_USERS","").split(",") if u.strip()]
if allow and actor not in allow:
    sys.exit(0)

body = ev["comment"]["body"]
issue = ev.get("issue") or ev.get("pull_request") or {}
issue_number = issue.get("number")

# รูปแบบคำสั่ง: /bot write PATH  + code block เนื้อหา
m = re.match(r"(?is)^/bot\s+(\w+)(?:\s+([^\n`]+))?", body.strip())
if not m:
    sys.exit(0)
cmd, arg = m.group(1).lower(), (m.group(2) or "").strip()
blocks = re.findall(r"(?is)```(?:\w+)?\n(.*?)```", body)
code = blocks[0] if blocks else ""

def git(*a): sh("git", *a)

branch = f"bot/issue-{issue_number}-{int(time.time())}"
git("checkout", "-b", branch)

changed = False
if cmd in ("write", "append"):
    if not arg:
        sys.exit("path required")
    p = pathlib.Path(arg)
    p.parent.mkdir(parents=True, exist_ok=True)
    if cmd == "write":
        p.write_text(code, encoding="utf-8")
    else:
        with open(p, "a", encoding="utf-8") as w:
            w.write(code)
    git("add", str(p))
    changed = True
elif cmd == "patch":
    if not code.strip().startswith("diff"):
        sys.exit("need unified diff")
    tmp = pathlib.Path("bot.patch")
    tmp.write_text(code, encoding="utf-8")
    try:
        git("apply", "-p0", "--whitespace=fix", str(tmp))
    except subprocess.CalledProcessError:
        git("apply", "-p1", "--whitespace=fix", str(tmp))
    git("add", "-A")
    changed = True
else:
    sys.exit(0)

if not changed:
    sys.exit(0)

git("commit", "-m", f"bot: {cmd} by @{actor} on #{issue_number}")
git("push", "-u", "origin", branch)

repo = ev["repository"]["full_name"]
base = ev["repository"]["default_branch"]

req = urllib.request.Request(
    f"https://api.github.com/repos/{repo}/pulls",
    data=json.dumps({
        "title": f"bot: {cmd} by @{actor} on #{issue_number}",
        "head": branch,
        "base": base,
        "body": body
    }).encode(),
    headers={
        "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "lbot-bot"
    },
    method="POST"
)
urllib.request.urlopen(req).read()
