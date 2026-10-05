"""Write the pip half of a lock: everything pip (not conda) installed.

    python lock_pip.py pip-compiled.txt locks/<env>

writes  locks/<env>-pip-compiled.txt   name==version, rebuilt w/o isolation
        locks/<env>-pip.txt            name==version or name @ vcs@commit
Editable installs (the own packages) are left out: they track the checkouts.
"""
import json
import re
import sys
from importlib.metadata import distributions


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_names(path):
    names = set()
    for line in open(path):
        line = line.split("#")[0].strip()
        if line:
            names.add(norm(re.split(r"[\s<>=!~;@\[]", line, maxsplit=1)[0]))
    return names


def main(compiled_txt, prefix):
    compiled = requirement_names(compiled_txt)
    out_compiled, out_other = [], []
    for dist in distributions():
        if (dist.read_text("INSTALLER") or "").strip() != "pip":
            continue
        name, version = dist.metadata["Name"], dist.version
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        if direct.get("dir_info", {}).get("editable"):
            continue
        vcs = direct.get("vcs_info")
        line = (f"{name} @ {vcs['vcs']}+{direct['url']}@{vcs['commit_id']}"
                if vcs else f"{name}=={version}")
        (out_compiled if norm(name) in compiled else out_other).append(line)
    for suffix, lines in (("-pip-compiled.txt", out_compiled), ("-pip.txt", out_other)):
        with open(prefix + suffix, "w") as f:
            f.write("\n".join(sorted(lines, key=str.lower)) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
