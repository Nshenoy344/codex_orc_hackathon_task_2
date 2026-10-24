"""Read-only integrity verification; opens no hardware and changes no source."""
from pathlib import Path
import hashlib,json,sys,os
root=Path(__file__).resolve().parents[1];bad=[];count=0
for row in (root/'manifest.sha256').read_text().splitlines():
 expected,name=row.split('  ',1);p=root/name
 if not p.is_file()or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:bad.append(name)
 count+=1
for item in json.loads((root/'symlinks.json').read_text()):
 p=root/item['path']
 if not p.is_symlink()or os.readlink(p)!=item['target']or not p.exists():bad.append(item['path'])
print(json.dumps({'checked_files':count,'mismatches':bad,'passed':not bad},indent=2))
sys.exit(1 if bad else 0)
