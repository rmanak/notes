#!/usr/bin/env python3
import os
import re
import sys
import subprocess

from datetime import datetime

def render_date(post_path: str) -> str:
    try:
        modified_time = os.path.getmtime(post_path)
        return datetime.fromtimestamp(modified_time).strftime("%a %b %d %Y")
    except OSError:
        return ""

def file_to_text(fname: str) -> str:
    try:
        with open(fname, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError as e:
        raise SystemExit(f"cannot open {fname}: {e}")


def run_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, text=True)
        return out.rstrip("\n")
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"command failed: {cmd}\n{e}")


def sub_literal(pattern: str, repl_text: str, text: str, flags: int = 0, count: int = 0) -> str:
    # Safe insertion of arbitrary text (repl_text may contain backslashes).
    rx = re.compile(pattern, flags)
    return rx.sub(lambda m: repl_text, text, count=count)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("\n  Usage: xxx template_file post_file\n\n")

    # hyper codes
    phc = "JDHALOWPFHSNC"            # post
    dhc = "ASDLKJFHG"                # date
    shc = "KJDSADPFP"                # sidebar
    fhc = "KDLPQHFZVD"               # footbar
    thc = "OAJDLASKJDHGPDH"          # title
    plhc = "IJDWONSADOHPDKLJASD"     # permanent link

    template_file = argv[1]
    post_file = argv[2]

    # Perl: `basename -s .w $post_file` then add .txt
    post_filetmp = os.path.splitext(os.path.basename(post_file))[0]
    post_filet = post_filetmp + ".txt"

    TEMPTXT = file_to_text(template_file)
    POSTTXT = file_to_text(post_file)
    SIDETXT = ""
    FOOTBARTXT = ""

    # date from helper script
    date = render_date(post_filet)
    # date = run_cmd(["./bin/mydate.py", post_filet])

    # title extraction / removal
    title = ""
    m = re.search(r"(title\s*:=\s*{\s*)([^}]*?)(\s*})", POSTTXT, flags=re.S)
    if m:
        title = m.group(2)
        POSTTXT = re.sub(r"title\s*:=\s*{[^}]*}", "", POSTTXT, flags=re.S)
    else:
        m2 = re.search(r"(<h1>\s*)([^:]*)(:*)(\s*</h1>)", POSTTXT, flags=re.S)
        if m2:
            title = m2.group(2)

    # author extraction / removal
    author = ""
    m = re.search(r"(author\s*:=\s*{\s*)([^}]*?)(\s*})", POSTTXT, flags=re.S)
    if m:
        author = m.group(2)
        POSTTXT = re.sub(r"author\s*:=\s*{[^}]*}", "", POSTTXT, flags=re.S)

    # keywords extraction / removal
    keywords = ""
    m = re.search(r"(keywords\s*:=\s*{\s*)([^}]*?)(\s*})", POSTTXT, flags=re.S)
    if m:
        keywords = m.group(2)
        POSTTXT = re.sub(r"keywords\s*:=\s*{[^}]*}", "", POSTTXT, flags=re.S)

    # description extraction / removal
    description = ""
    m = re.search(r"(description\s*:=\s*{\s*)([^}]*?)(\s*})", POSTTXT, flags=re.S)
    if m:
        description = m.group(2)
        POSTTXT = re.sub(r"description\s*:=\s*{[^}]*}", "", POSTTXT, flags=re.S)

    # update <title>...</title>
    if title != "":
        TEMPTXT = re.sub(
            r"<title>.*</title>",
            lambda _m: f"<title>{title}</title>",
            TEMPTXT,
            flags=re.S,
        )

    # update meta tags (safe insertion)
    if keywords != "":
        rx = re.compile(r'(<meta name="keywords" content=")([^"]*)("\s*/>)', flags=re.S)
        TEMPTXT = rx.sub(lambda m: m.group(1) + keywords + m.group(3), TEMPTXT)

    if author != "":
        rx = re.compile(r'(<meta name="author" content=")([^"]*)("\s*/>)', flags=re.S)
        TEMPTXT = rx.sub(lambda m: m.group(1) + author + m.group(3), TEMPTXT)

    if description != "":
        rx = re.compile(r'(<meta name="description" content=")([^"]*)("\s*/>)', flags=re.S)
        TEMPTXT = rx.sub(lambda m: m.group(1) + description + m.group(3), TEMPTXT)

    # syntactic shorthands
    dl_syn = "[dl]"
    dl_op = '<img src="../img/dl.svg" />'

    ext_lnk_syn = "[>]"
    ext_lnk_op = '<img style="margin-left:1px;" src="../img/external-link.svg" alt="" align="bottom" />'

    toc_syn = "[TOC]"
    toc_op = '<div class="table_of_contents"></div>'

    # Creating references:
    num_ref = 0
    ref_txt = ""
    is_there_ref = 0
    ref_nums: dict[str, int] = {}

    token_pat = re.compile(r"(%%\s*)(\w+.*?)(\s*%%)")
    while True:
        m = token_pat.search(POSTTXT)
        if not m:
            break

        is_there_ref = 1
        lnk_nm = m.group(2)

        if lnk_nm not in ref_nums:
            num_ref += 1
            ref_nums[lnk_nm] = num_ref

            POSTTXT = re.sub(
                rf"(%%\s*){re.escape(lnk_nm)}(\s*%%)",
                f'<sup><a href="#ref{lnk_nm}" id="cnt{lnk_nm}">[{num_ref}]</a></sup>',
                POSTTXT,
                count=1,
            )

            # Find and remove the definition block: { lnk_nm : ... }
            ref_def_pat = re.compile(r"(\{\s*" + re.escape(lnk_nm) + r"\s*:\s*)([^\}]*)(\})", flags=re.S)
            mdef = ref_def_pat.search(POSTTXT)
            if mdef:
                each_ref = f'<a href="#cnt{lnk_nm}" id="ref{lnk_nm}">[{num_ref}]</a>: {mdef.group(2)}'
                ref_txt += f"<p>{each_ref}</p>\n"
                POSTTXT = ref_def_pat.sub("", POSTTXT)
        else:
            num = ref_nums[lnk_nm]
            POSTTXT = re.sub(
                rf"(%%\s*){re.escape(lnk_nm)}(\s*%%)",
                f'<sup><a href="#ref{lnk_nm}" id="cnt{lnk_nm}">[{num}]</a></sup>',
                POSTTXT,
                count=1,
            )

    if is_there_ref == 1:
        POSTTXT = POSTTXT + "<hr />" + "<h2>References</h2>" + ref_txt

    # Fill template placeholders (use .replace or safe literal sub for arbitrary content)
    TEMPTXT = TEMPTXT.replace(dhc, date)

    # These may contain backslashes -> must be safe:
    TEMPTXT = sub_literal(re.escape(phc), POSTTXT, TEMPTXT)
    TEMPTXT = sub_literal(re.escape(shc), SIDETXT, TEMPTXT)
    TEMPTXT = sub_literal(re.escape(fhc), FOOTBARTXT, TEMPTXT)
    TEMPTXT = sub_literal(re.escape(thc), title, TEMPTXT)

    # Simple literal shorthands
    TEMPTXT = TEMPTXT.replace(ext_lnk_syn, ext_lnk_op)
    TEMPTXT = TEMPTXT.replace(dl_syn, dl_op)
    TEMPTXT = TEMPTXT.replace(toc_syn, toc_op)

    # permalink slug
    title_slug = re.sub(r"\s+", " ", title).strip()
    title_slug = title_slug.lower().replace(" ", "-")
    TEMPTXT = TEMPTXT.replace(plhc, title_slug)

    # image expansions
    img_op1 = '<div style="max-width:350px; width:100%; padding:2px; height:auto; text-align:justify; border: solid #BBBBBB 1px;float:right; margin-left:8px;"><img style="display:block; height:auto; width:100%; max-width:320px; margin-left:auto; margin-right:auto;" src="'
    img_op2 = '" alt="'
    img_op3 = '" border="0" /><hr />'
    img_op3noline = '" border="0" />'
    img_op4 = "</div>"

    TEMPTXT = re.sub(r"({{}}\(\()(.*?)(\)\))", rf"{img_op1}\2{img_op2}{img_op3noline}{img_op4}", TEMPTXT, flags=re.S)
    TEMPTXT = re.sub(r"({{)(.*?)(}}\(\()(.*?)(\)\))", rf"{img_op1}\4{img_op2}{img_op3}\2{img_op4}", TEMPTXT, flags=re.S)

    imgc_op1 = '<div style="margin-left:auto; margin-right:auto; padding:3px; text-align:justify; border:solid #BBBBBB 1px; height:auto; width: 100%; max-width:500px;"><img style="display:block; height:auto; width:100%; max-width:470px; margin-left:auto; margin-right:auto;" src="'
    imgc_op2 = '" alt="'
    imgc_op3 = '" border="0" /><hr />'
    imgc_op3noline = '" border="0" />'
    imgc_op4 = "</div>"

    TEMPTXT = re.sub(r"({{}}\[\[)(.*?)(\]\])", rf"{imgc_op1}\2{imgc_op2}{imgc_op3noline}{imgc_op4}", TEMPTXT, flags=re.S)
    TEMPTXT = re.sub(r"({{)(.*?)(}}\[\[)(.*?)(\]\])", rf"{imgc_op1}\4{imgc_op2}{imgc_op3}\2{imgc_op4}", TEMPTXT, flags=re.S)

    # <R>...</R> block
    r_op1 = '<div style="max-width:350px; width:100%; padding:2px; height:auto; text-align:justify; border: solid #BBBBBB 1px;float:right; margin-left:8px;">'
    r_op2 = "</div>"
    TEMPTXT = re.sub(r"(<R>)(.*?)(</R>)", rf"{r_op1}\2{r_op2}", TEMPTXT, flags=re.S)

    sys.stdout.write(TEMPTXT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
