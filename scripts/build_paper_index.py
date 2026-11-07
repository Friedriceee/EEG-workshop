#!/usr/bin/env python3
"""检查 papers/*.md 的 frontmatter 并生成 papers/INDEX.md。只用标准库。

用法:
    python scripts/build_paper_index.py          # 校验 + 重新生成 INDEX.md
    python scripts/build_paper_index.py --check  # 只校验,不写文件(CI 用)
"""
import re
import sys
from pathlib import Path

PAPERS = Path(__file__).resolve().parent.parent / "papers"
REQUIRED = ["title", "authors", "year", "venue", "link", "added_by", "date_added", "tags", "status", "relevance"]
STATUS = {"to-read", "skimmed", "read", "reproduced"}
RELEVANCE = {"high", "medium", "low"}
NAME_RE = re.compile(r"^\d{4}_[a-z0-9-]+_[a-z0-9_-]+\.md$")
PLACEHOLDERS = {"论文完整标题", "你的名字", "第一作者 et al.", "期刊/会议名 或 arXiv"}


def parse(path):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        raise ValueError("缺少 frontmatter(文件开头需要 --- ... --- 块)")
    meta = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"无法解析这一行: {line!r}")
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip().strip('"')
    return meta


def validate(path, meta):
    errs = []
    if not NAME_RE.match(path.name):
        errs.append("文件名应为 <年份>_<作者姓>_<关键词>.md,小写,如 2020_he_euclidean-alignment.md")
    for k in REQUIRED:
        if not meta.get(k):
            errs.append(f"frontmatter 缺少字段: {k}")
    if meta.get("title") in PLACEHOLDERS or meta.get("added_by") in PLACEHOLDERS:
        errs.append("frontmatter 里还是模板占位文字,请改成真实内容")
    if meta.get("year") and not re.fullmatch(r"(19|20)\d{2}", meta["year"]):
        errs.append(f"year 不是四位年份: {meta['year']}")
    if meta.get("link") and not meta["link"].startswith(("http://", "https://")):
        errs.append("link 需要以 http(s):// 开头")
    if "xxxx" in meta.get("link", ""):
        errs.append("link 还是模板占位链接,请改成真实链接")
    if meta.get("status") and meta["status"] not in STATUS:
        errs.append(f"status 只能是 {sorted(STATUS)}")
    if meta.get("relevance") and meta["relevance"] not in RELEVANCE:
        errs.append(f"relevance 只能是 {sorted(RELEVANCE)}")
    if meta.get("tags") and not re.fullmatch(r"\[.*\]", meta["tags"]):
        errs.append("tags 请写成 [a, b] 形式")
    if meta.get("date_added") and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", meta["date_added"]):
        errs.append("date_added 请写成 YYYY-MM-DD")
    return errs


def esc(s):
    return s.replace("|", "\\|")


def main():
    check_only = "--check" in sys.argv
    files = sorted(p for p in PAPERS.glob("*.md") if not p.name.startswith("_") and p.name not in {"README.md", "INDEX.md"})
    rows, failed = [], False
    for p in files:
        try:
            meta = parse(p)
            errs = validate(p, meta)
        except ValueError as e:
            meta, errs = {}, [str(e)]
        for e in errs:
            print(f"✗ {p.name}: {e}")
            failed = True
        if not errs:
            rows.append((p, meta))
    if failed:
        print("\n格式有问题,请修改后重试。")
        sys.exit(1)

    order = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda r: (order[r[1]["relevance"]], -int(r[1]["year"]), r[0].name))
    lines = [
        "# 论文索引",
        "",
        "> 本文件由 `scripts/build_paper_index.py` 自动生成,**请勿手改**。按相关度、年份排序。",
        "",
        f"共 {len(rows)} 篇。",
        "",
        "| 相关度 | 年份 | 论文 | 会议/期刊 | 标签 | 状态 | 添加人 | 笔记 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p, m in rows:
        tags = ", ".join(t.strip() for t in m["tags"].strip("[]").split(",") if t.strip())
        lines.append(
            f"| {m['relevance']} | {m['year']} | [{esc(m['title'])}]({m['link']}) — {esc(m['authors'])} "
            f"| {esc(m['venue'])} | {esc(tags)} | {m['status']} | {esc(m['added_by'])} | [笔记]({p.name}) |"
        )
    out = "\n".join(lines) + "\n"
    if check_only:
        print(f"✓ {len(rows)} 篇论文格式正确")
        return
    (PAPERS / "INDEX.md").write_text(out, encoding="utf-8")
    print(f"✓ 已生成 papers/INDEX.md({len(rows)} 篇)")


if __name__ == "__main__":
    main()
