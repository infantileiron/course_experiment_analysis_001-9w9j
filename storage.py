import hashlib, json, os, uuid
from pathlib import Path
from datetime import datetime, timezone

class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
    def put(self, kind, payload, identifier=None):
        identifier = identifier or uuid.uuid4().hex[:16]
        directory = self.root / kind
        directory.mkdir(exist_ok=True)
        target = directory / (identifier + '.json')
        if target.exists():
            raise ValueError('Immutable record already exists')
        record = dict(payload, id=identifier, created_at=datetime.now(timezone.utc).isoformat())
        text = json.dumps(record, ensure_ascii=False, allow_nan=False)
        temp = target.with_suffix('.tmp')
        temp.write_text(text, encoding='utf-8')
        os.replace(temp, target)
        return record
    def get(self, kind, identifier):
        if not identifier or any(c not in '0123456789abcdefghijklmnopqrstuvwxyz_-' for c in identifier):
            raise ValueError('Invalid identifier')
        path = self.root / kind / (identifier + '.json')
        if not path.exists():
            raise ValueError('Record not found')
        return json.loads(path.read_text(encoding='utf-8'))
    def list(self, kind):
        directory = self.root / kind
        return sorted([json.loads(p.read_text(encoding='utf-8')) for p in directory.glob('*.json')], key=lambda row: row['created_at'], reverse=True) if directory.exists() else []

def fingerprint(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
