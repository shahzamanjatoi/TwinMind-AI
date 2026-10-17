import re
from datetime import datetime
from pathlib import Path

path = Path("app.py")
raw = path.read_bytes()
bom = raw.startswith(b"\xef\xbb\xbf")
text = raw.decode("utf-8-sig")
nl = "\r\n" if "\r\n" in text else "\n"
done = []

# 1. remove the empty sidebar slot
text, n = re.subn(r"^nav_slot = st\.sidebar\.container\(\)[ \t]*\r?\n(?:[ \t]*\r?\n)?", "", text, flags=re.M)
if n:
    done.append("removed the sidebar slot")

# 2. section menu on the main page, as a horizontal row
pattern = r"^with nav_slot:[ \t]*\r?\n[ \t]+active_section = st\.radio\(.*\)[ \t]*(?=\r?$)"
new_radio = 'active_section = st.radio("Section", TAB_NAMES, key="section_nav", horizontal=True, label_visibility="collapsed")'
text, n = re.subn(pattern, lambda m: new_radio, text, flags=re.M)
if n:
    done.append("moved the section menu to the main page")

# 3. sidebar caption wording
old = ('"The GPU is used by live screening in AMD GPU. The Assistant tab "' + nl +
       '        "reads the saved AMD screening reports."')
new = ('"The GPU is used by the live screening in the AMD GPU section. The "' + nl +
       '        "Assistant section reads the saved AMD screening reports."')
if text.count(old) == 1:
    text = text.replace(old, new)
    done.append("fixed the sidebar caption wording")

if "nav_slot" in text:
    raise SystemExit("Stopped: 'nav_slot' is still in app.py, so nothing was changed. Paste the lines around it here.")
if not done:
    raise SystemExit("Nothing to change: app.py already has these edits.")

compile(text, "app.py", "exec")
backup = path.with_name(f"app.py.bak-final-{datetime.now():%Y%m%d-%H%M%S}")
backup.write_bytes(raw)
path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
print("Done:", "; ".join(done))
print("Backup:", backup.name)
