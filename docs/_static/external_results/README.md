# External benchmark snapshots

This directory contains only reviewed aggregate outputs intended for the public
documentation. Raw datasets, extracted caches, embeddings, row-level scores, and
full experiment workspaces must not be committed here.

Snapshots are created with `scripts/publish_external_snapshot.py`; each snapshot
has a `snapshot.json` provenance record and cryptographic digests for its files.
