"""Validate numerical evidence separately from model admission and human review."""
import hashlib
import json
from pathlib import Path
from .dsl import ValidationError
from .model import ROOT
from .structure import check


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_record(path, root=ROOT, current=False):
    root = Path(root).resolve()
    record = json.loads(Path(path).read_text())
    schema = {'numerical-evidence':'evidence.schema.json', 'source-review':'review.schema.json'}.get(record.get('kind'))
    if not schema:
        raise ValidationError('Unsupported evidence/review record kind')
    check(record, json.loads((ROOT / 'schemas' / schema).read_text()))
    if record['kind'] == 'source-review':
        if record['status'] != 'pending' and (not record['independent'] or not record.get('reviewer') or not record.get('reviewed_at')):
            raise ValidationError('Completed review requires independent reviewer identity and date')
        if record['status'] == 'pending' and not record['unresolved']:
            raise ValidationError('Pending review must identify unresolved work')
        if record['status'] == 'accepted' and record['unresolved']:
            raise ValidationError('Accepted review cannot have unresolved questions')
        if not (root / record['dossier']).is_file():
            raise ValidationError('Review dossier is missing')
    seen = set()
    for item in record.get('artifacts', []) + (record.get('source_files', []) if current else []):
        target = (root / item['path']).resolve()
        if not target.is_relative_to(root) or target == root or item['path'] in seen:
            raise ValidationError('Artifact paths must be unique files within the repository')
        seen.add(item['path'])
        if not target.is_file() or sha256(target) != item['sha256']:
            raise ValidationError('Evidence artifact missing or changed: ' + item['path'])
    if current:
        model = root / 'models' / (record['model']['id'] + '.json')
        if not model.is_file() or sha256(model) != record['model']['definition_sha256']:
            raise ValidationError('Evidence does not match the current model definition')
    # An envelope establishes integrity and attribution, not scientific validity.
    # The named checker owns report-specific numerical assertions.
    return record


# Historical source receipt verification preserves the public record API above.
def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_receipt(root, document, original_path, expected):
    """Return a verified historical snapshot or unchanged live file; else fail.

    ``document`` is the repository-relative immutable receipt document. A
    sidecar binds its exact original bytes and each redirected path/hash pair.
    No expected hash is recomputed or replaced, and an archive never falls back
    to the live source when corrupt or missing.
    """
    root = Path(root).resolve()
    document = Path(document)

    def within(relative):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe evidence path')
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError('evidence path escapes repository')
        return resolved

    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('invalid expected evidence hash')
    doc = within(document)
    def contains_receipt(node):
        if isinstance(node, dict):
            if node.get(original_path) == expected:
                return True
            if node.get('path') == original_path and any(node.get(key) == expected for key in ('sha256', 'hash')):
                return True
            return any(contains_receipt(value) for value in node.values())
        return isinstance(node, list) and any(contains_receipt(value) for value in node)
    if not contains_receipt(json.loads(doc.read_text())):
        raise ValueError('source path/hash pair absent from original receipt document')
    selected = within(original_path)
    sidecar = doc.parent / 'source-archive-map.json'
    if sidecar.exists():
        mapping = json.loads(sidecar.read_text())
        if mapping.get('version') != 'immutable-source-archive-map-0.1':
            raise ValueError('unsupported source archive map')
        docpin = mapping.get('documents', {}).get(document.as_posix())
        if docpin is None or _sha(doc) != docpin:
            raise ValueError('original evidence document changed after archival')
        entries = mapping.get('entries')
        if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
            raise ValueError('invalid archive entries')
        matches = [entry for entry in entries
                   if entry.get('original_path') == original_path and entry.get('sha256') == expected]
        if len(matches) > 1:
            raise ValueError('ambiguous source archive map')
        if matches:
            relative = Path('catalogue/source-snapshots') / expected / original_path
            if matches[0].get('archived_path') != relative.as_posix():
                raise ValueError('invalid archive location')
            selected = within(relative)
    try:
        actual = _sha(selected)
    except OSError as exc:
        raise ValueError('missing evidence source: ' + str(selected)) from exc
    if actual != expected:
        raise ValueError('evidence source hash mismatch: ' + str(selected))
    return selected


def verify_manifest_files(root, document):
    """Check the existing dependencies/files union using original receipt pins."""
    root = Path(root).resolve()
    document = Path(document)
    if document.is_absolute() or '..' in document.parts or not (root/document).resolve().is_relative_to(root):
        raise ValueError('unsafe evidence document path')
    manifest = json.loads((root / document).read_text())
    for path, expected in (manifest.get('dependencies', {}) | manifest['files']).items():
        checked_receipt(root, document, path, expected)
    return manifest
