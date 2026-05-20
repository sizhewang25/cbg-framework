"""Integration tests for the v2 CBGModel pipeline.

These tests exercise full LTD → MTL → CTR pipelines across every valid
family pairing, plus the composition-time guards (family validation,
fallback behavior, registry-driven construction).

Per-stage unit tests live in scripts/framework/v2/{ltd,mtl,ctr}/tests/.
"""
