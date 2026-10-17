"""Rebuild checkpoints must survive failed files and new process instances."""

from pathlib import Path
from unittest.mock import patch

import pytest

from memweave import MemWeave
from memweave.exceptions import EmbeddingError, StorageError
from tests.unit.test_store import MockEmbeddingProvider, _make_config, _write_md


@pytest.mark.parametrize("retry_force", [False, True])
async def test_rebuild_resumes_without_dropping_completed_vectors(
    tmp_path: Path, retry_force: bool
):
    for name in ("a", "b", "c"):
        _write_md(tmp_path, f"memory/{name}.md", f"Unique document {name}")
    cfg = _make_config(tmp_path)
    mem = MemWeave(cfg, embedding_provider=MockEmbeddingProvider())
    await mem.index()
    original = mem._index_file

    async def interrupt(path, workspace, *, force=False):
        result = await original(path, workspace, force=force)
        if path.name == "b.md":
            raise RuntimeError("simulated interruption after file writes")
        return result

    with patch.object(mem, "_index_file", side_effect=interrupt):
        with pytest.raises(RuntimeError, match="simulated interruption"):
            await mem.index(force=True)
    await mem.close()

    resumed = MemWeave(cfg, embedding_provider=MockEmbeddingProvider())
    async with resumed:
        original = resumed._index_file
        with patch.object(resumed, "_index_file", wraps=original) as indexed:
            result = await resumed.index(force=retry_force)
        assert result.files_indexed == 2
        assert result.files_skipped == 1
        assert [call.kwargs["force"] for call in indexed.call_args_list] == [False, True, True]
        assert resumed._db is not None
        rows = await (await resumed._db.execute("SELECT count(*) FROM chunks_vec")).fetchone()
        assert rows[0] == 3
        assert await resumed._store.get_meta("memory_index_rebuild_v1") is None
        # Another explicit force after success is a new generation.
        assert (await resumed.index(force=True)).files_indexed == 3


async def test_failed_vector_write_rolls_back_the_file(tmp_path: Path):
    _write_md(tmp_path, "memory/a.md", "An original document")
    mem = MemWeave(_make_config(tmp_path), embedding_provider=MockEmbeddingProvider())
    async with mem:
        with patch.object(mem, "_upsert_vec", side_effect=StorageError("disk full")):
            with pytest.raises(StorageError, match="disk full"):
                await mem.index(force=True)
        assert (await mem.files()) == []
        assert (await mem.index(force=True)).files_indexed == 1


async def test_changed_configuration_restarts_pending_generation(tmp_path: Path):
    _write_md(tmp_path, "memory/a.md", "One document")
    cfg = _make_config(tmp_path)
    mem = MemWeave(cfg, embedding_provider=MockEmbeddingProvider())
    async with mem:
        with patch.object(mem, "_index_file", side_effect=RuntimeError("interrupted")):
            with pytest.raises(RuntimeError):
                await mem.index(force=True)
    cfg.embedding.model = "different-model"
    async with MemWeave(cfg, embedding_provider=MockEmbeddingProvider(dims=4)) as new:
        assert (await new.index()).files_indexed == 1
        row = await (await new._db.execute("SELECT length(embedding) FROM chunks_vec")).fetchone()
        assert row[0] == 16


async def test_large_file_retains_embedding_batches_without_publishing_partial_text(tmp_path: Path):
    path = _write_md(tmp_path, "memory/large.md", "Original searchable text")
    cfg = _make_config(tmp_path)
    cfg.embedding.batch_size = 1
    cfg.chunking.tokens = 8
    cfg.chunking.overlap = 0
    provider = MockEmbeddingProvider()
    mem = MemWeave(cfg, embedding_provider=provider)
    await mem.index()
    old_hash = (await mem.files())[0].hash
    path.write_text(
        "\n\n".join(f"Section {i} contains different information {i}" for i in range(6))
    )
    original = provider.embed_batch
    cached_texts = []

    async def interrupted_batch(texts):
        if cached_texts:
            raise EmbeddingError("backend unavailable")
        cached_texts.extend(texts)
        return await original(texts)

    with patch.object(provider, "embed_batch", side_effect=interrupted_batch):
        with patch.object(mem, "_embed_with_halving", return_value=None):
            with pytest.raises(EmbeddingError, match="not checkpointed"):
                await mem.index(force=True)
    # Cached API work is durable, but the file is still the previous complete version.
    assert (await mem.files())[0].hash == old_hash
    rows = await (await mem._db.execute("SELECT text FROM chunks_fts")).fetchall()
    assert "Original searchable text" in " ".join(row[0] for row in rows)
    await mem.close()

    resumed_provider = MockEmbeddingProvider()
    async with MemWeave(cfg, embedding_provider=resumed_provider) as resumed:
        result = await resumed.index(force=True)
        assert result.files_indexed == 1
        assert result.embeddings_cached >= 1
        assert all(
            text not in cached_texts
            for batch in resumed_provider.embed_batch_calls
            for text in batch
        )
        assert (await resumed.files())[0].hash != old_hash


async def test_vector_replacement_updates_existing_id_atomically(tmp_path: Path):
    import struct

    _write_md(tmp_path, "memory/a.md", "A document with an existing vector")
    async with MemWeave(_make_config(tmp_path), embedding_provider=MockEmbeddingProvider()) as mem:
        await mem.index()
        row = await (await mem._db.execute("SELECT id, embedding FROM chunks_vec")).fetchone()
        replacement = [0.25] * (len(row[1]) // 4)
        await mem._upsert_vec(row[0], replacement)
        updated = await (
            await mem._db.execute("SELECT embedding FROM chunks_vec WHERE id=?", [row[0]])
        ).fetchone()
        assert list(struct.unpack(f"{len(replacement)}f", updated[0])) == replacement
        await mem._store.rollback()
        restored = await (
            await mem._db.execute("SELECT embedding FROM chunks_vec WHERE id=?", [row[0]])
        ).fetchone()
        assert restored[0] == row[1]
