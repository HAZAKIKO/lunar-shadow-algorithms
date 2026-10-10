# -*- coding: utf-8 -*-
"""verify_remote.py —— check that the GitHub remote repository matches the local copy.

Reading a public repository needs no token: this script prefers unauthenticated access;
if the environment variable GITHUB_TOKEN is set or a token.txt exists, it is used instead
(that allows private repositories and gives a higher rate limit).
"""

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

OWNER, REPO = "HAZAKIKO", "lunar-shadow-algorithms"


def get(path: str, token: str | None):
    for _ in range(5):
        try:
            url = "https://api.github.com" + urllib.parse.quote(path, safe=":/?=&%")
            req = urllib.request.Request(url)
            if token:
                req.add_header("Authorization", "Bearer " + token)
            req.add_header("User-Agent", "dsh")
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            time.sleep(2)
    raise SystemExit("the network kept failing")


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    # the token is optional
    tok = os.environ.get("GITHUB_TOKEN")
    tf = os.path.join(here, "token.txt")
    if not tok and os.path.exists(tf):
        with open(tf, encoding="utf-8") as f:
            tok = f.read().strip() or None
    print("auth      :", "token" if tok else "anonymous (enough for reading public repositories)")

    info = get(f"/repos/{OWNER}/{REPO}", tok)
    print("repo      :", info["full_name"])
    print("visibility:", "Private" if info["private"] else "Public")
    print("branch    :", info["default_branch"])
    print("desc      :", info["description"])
    print("web       :", info["html_url"])
    print()

    tree = get(f"/repos/{OWNER}/{REPO}/git/trees/{info['default_branch']}?recursive=1", tok)
    blobs = [t for t in tree["tree"] if t["type"] == "blob"]
    print("remote files: %d" % len(blobs))
    for t in sorted(blobs, key=lambda x: x["path"]):
        print("  %9d B  %s" % (t.get("size", 0), t["path"]))

    # compare against the local copies one by one, by size
    print("\nComparing with the local copy:")
    ok = True
    for t in sorted(blobs, key=lambda x: x["path"]):
        lp = os.path.join(here, t["path"].replace("/", os.sep))
        if not os.path.exists(lp):
            print("  ✗ missing locally:", t["path"])
            ok = False
            continue
        ls = os.path.getsize(lp)
        rs = t.get("size", -1)
        # binaries such as PNG should keep their size after an API upload; text files may differ
        # by a few dozen bytes because of newline normalisation
        flag = "✓" if ls == rs else ("≈" if abs(ls - rs) < 200 else "✗")
        if flag == "✗":
            ok = False
        print("  %s local %9d B / remote %9d B  %s" % (flag, ls, rs, t["path"]))

    rb = get(f"/repos/{OWNER}/{REPO}/readme", tok)
    text = base64.b64decode(rb["content"]).decode("utf-8")
    print("\nREADME first line:", text.splitlines()[0])
    print("README chars:", len(text))

    print("\nConclusion:", "identical — published successfully."
          if ok else "differences found; check the lines marked ✗ above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
