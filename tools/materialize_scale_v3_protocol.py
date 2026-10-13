#!/usr/bin/env python3
"""Resolve portable scale-v3 protocol templates and verify required artifacts."""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

DATA_TOKEN = chr(36) + "{CRAFT_DATA_ROOT}"
REPO_TOKEN = chr(36) + "{REPO_ROOT}"


def expand(value, data_root, repo_root):
    if isinstance(value, str):
        return value.replace(DATA_TOKEN, str(data_root)).replace(REPO_TOKEN, str(repo_root))
    if isinstance(value, list):
        return [expand(item, data_root, repo_root) for item in value]
    if isinstance(value, dict):
        return {expand(key, data_root, repo_root): expand(item, data_root, repo_root)
                for key, item in value.items()}
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize(template, data_root, repo_root, output):
    protocol = expand(json.loads(Path(template).read_text()),
                      Path(data_root).resolve(), Path(repo_root).resolve())
    checkpoint = Path(protocol["source_checkpoint"])
    if not checkpoint.is_dir():
        raise FileNotFoundError("Required checkpoint is missing: " + str(checkpoint))
    for name in ("agent.pkl", "replay.pkl", "runtime.pkl", "step.pkl", "done"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError("Incomplete checkpoint: " + str(checkpoint / name))
    step = int(pickle.loads((checkpoint / "step.pkl").read_bytes()))
    if step != int(protocol["source_step"]):
        raise ValueError("Checkpoint step mismatch: %s != %s" % (step, protocol["source_step"]))
    if sha256(checkpoint / "agent.pkl") != protocol["source_agent_sha256"]:
        raise ValueError("Checkpoint agent SHA256 mismatch")
    config = Path(protocol["source_config"])
    if sha256(config) != protocol["source_config_sha256"]:
        raise ValueError("Source config SHA256 mismatch: " + str(config))
    bank = protocol["bank"]
    for audit_name in bank["audits"]:
        audit_path = Path(audit_name)
        if sha256(audit_path) != bank["audit_sha256"][str(audit_path)]:
            raise ValueError("Bank audit SHA256 mismatch: " + str(audit_path))
        for row in json.loads(audit_path.read_text())["rows"]:
            bank_file = Path(bank["directory"]) / row["bank_file"]
            if sha256(bank_file) != row["bank_sha256"]:
                raise ValueError("Missing or invalid replay bank file: " + str(bank_file))
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(protocol, indent=2) + "\n")
    temporary.replace(output)
    return protocol


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--data-root", default=os.environ.get("CRAFT_DATA_ROOT"))
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.data_root:
        parser.error("provide --data-root or set CRAFT_DATA_ROOT")
    try:
        protocol = materialize(args.template, args.data_root, args.repo_root, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "source_step": protocol["source_step"],
        "target": protocol["target"],
        "bank_kind": protocol["bank"]["kind"],
        "verified_inputs": True,
    }, indent=2))


if __name__ == "__main__":
    main()
