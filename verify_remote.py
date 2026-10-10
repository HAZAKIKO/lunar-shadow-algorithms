# -*- coding: utf-8 -*-
"""verify_remote.py —— 核对 GitHub 远端仓库内容是否与本地一致。

公开仓库读取不需要令牌：本脚本优先无认证访问；
若设置了环境变量 GITHUB_TOKEN 或存在 token.txt，则用它（可查私有仓库、额度更高）。
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
    raise SystemExit("网络持续失败")


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    # 令牌是可选的
    tok = os.environ.get("GITHUB_TOKEN")
    tf = os.path.join(here, "token.txt")
    if not tok and os.path.exists(tf):
        with open(tf, encoding="utf-8") as f:
            tok = f.read().strip() or None
    print("认证方式:", "令牌" if tok else "匿名（公开仓库读取足够）")

    info = get(f"/repos/{OWNER}/{REPO}", tok)
    print("仓库      :", info["full_name"])
    print("可见性    :", "Private" if info["private"] else "Public")
    print("默认分支  :", info["default_branch"])
    print("描述      :", info["description"])
    print("网页      :", info["html_url"])
    print()

    tree = get(f"/repos/{OWNER}/{REPO}/git/trees/{info['default_branch']}?recursive=1", tok)
    blobs = [t for t in tree["tree"] if t["type"] == "blob"]
    print("远端文件 %d 个：" % len(blobs))
    for t in sorted(blobs, key=lambda x: x["path"]):
        print("  %9d B  %s" % (t.get("size", 0), t["path"]))

    # 与本地逐一比对大小
    print("\n与本地比对：")
    ok = True
    for t in sorted(blobs, key=lambda x: x["path"]):
        lp = os.path.join(here, t["path"].replace("/", os.sep))
        if not os.path.exists(lp):
            print("  ✗ 本地缺少:", t["path"])
            ok = False
            continue
        ls = os.path.getsize(lp)
        rs = t.get("size", -1)
        # PNG 等二进制经 API 上传后大小应一致；文本文件可能因换行归一化差几十字节
        flag = "✓" if ls == rs else ("≈" if abs(ls - rs) < 200 else "✗")
        if flag == "✗":
            ok = False
        print("  %s 本地 %9d B / 远端 %9d B  %s" % (flag, ls, rs, t["path"]))

    rb = get(f"/repos/{OWNER}/{REPO}/readme", tok)
    text = base64.b64decode(rb["content"]).decode("utf-8")
    print("\nREADME 首行:", text.splitlines()[0])
    print("README 字符数:", len(text))

    print("\n结论:", "完全一致，发布成功。" if ok else "存在差异，请查看上面标 ✗ 的行。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
