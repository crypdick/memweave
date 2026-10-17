"""
tests/unit/test_storage.py — Unit tests for storage/schema.py, files.py, sqlite_store.py
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

from memweave.storage.files import (
    get_source_from_path,
    is_evergreen,
    is_memory_path,
    list_memory_files,
    relative_path,
)
from memweave.storage.schema import ensure_schema, get_schema_version
from memweave.storage.sqlite_store import SQLiteStore

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db():
    """In-memory SQLite database with schema applied."""
    async with aiosqlite.connect(":memory:") as conn:
        await ensure_schema(conn)
        await conn.commit()
        yield conn


@pytest_asyncio.fixture
async def store(db):
    """SQLiteStore wrapping in-memory database."""
    return SQLiteStore(db)


# ── schema tests ──────────────────────────────────────────────────────────────


class TestSchema:
    async def test_migrates_existing_fts_rows_without_reembedding(self):
        async with aiosqlite.connect(":memory:") as conn:
            await ensure_schema(conn)
            await conn.execute("DROP TABLE memweave_fts_lookup")
            await conn.execute("DROP TABLE index_rebuild_files")
            await conn.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
            await conn.execute(
                "INSERT INTO chunks_fts(text,id,path) VALUES ('original','a','old.md')"
            )
            await conn.commit()
            await ensure_schema(conn)
            store = SQLiteStore(conn)
            await store.upsert_fts("replacement", "a", "old.md", "memory", 1, 1)
            rows = await (await conn.execute("SELECT text FROM chunks_fts")).fetchall()
            assert rows == [("replacement",)]
            await store.delete_fts_by_path("old.md")
            assert await (await conn.execute("SELECT * FROM chunks_fts")).fetchall() == []

    @pytest.mark.asyncio
    async def test_schema_creates_meta_table(self, db):
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
        )
        row = await cursor.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_schema_creates_files_table(self, db):
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='files'"
        )
        assert await cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_schema_creates_chunks_table(self, db):
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        )
        assert await cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_schema_creates_fts_table(self, db):
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        )
        assert await cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_schema_creates_embedding_cache(self, db):
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_cache'"
        )
        assert await cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_schema_is_idempotent(self, db):
        """Calling ensure_schema twice should not raise."""
        await ensure_schema(db)  # second call
        await ensure_schema(db)  # third call

    @pytest.mark.asyncio
    async def test_schema_version(self, db):
        version = await get_schema_version(db)
        assert version == 2


# ── SQLiteStore meta tests ────────────────────────────────────────────────────


class TestSQLiteStoreMeta:
    @pytest.mark.asyncio
    async def test_set_and_get_meta(self, store, db):
        await store.set_meta("my_key", "my_value")
        await store.commit()
        value = await store.get_meta("my_key")
        assert value == "my_value"

    @pytest.mark.asyncio
    async def test_get_missing_meta(self, store):
        value = await store.get_meta("does_not_exist")
        assert value is None

    @pytest.mark.asyncio
    async def test_meta_overwrite(self, store, db):
        await store.set_meta("key", "v1")
        await store.set_meta("key", "v2")
        await store.commit()
        assert await store.get_meta("key") == "v2"

    @pytest.mark.asyncio
    async def test_get_all_meta(self, store, db):
        await store.set_meta("a", "1")
        await store.set_meta("b", "2")
        await store.commit()
        meta = await store.get_all_meta()
        assert meta["a"] == "1"
        assert meta["b"] == "2"


# ── SQLiteStore files tests ───────────────────────────────────────────────────


class TestSQLiteStoreFiles:
    @pytest.mark.asyncio
    async def test_upsert_and_get_file(self, store, db):
        await store.upsert_file("memory/test.md", "memory", "abc123", 1000.0, 512)
        await store.commit()
        record = await store.get_file("memory/test.md")
        assert record is not None
        assert record["path"] == "memory/test.md"
        assert record["hash"] == "abc123"
        assert record["size"] == 512

    @pytest.mark.asyncio
    async def test_get_nonexistent_file(self, store):
        assert await store.get_file("does/not/exist.md") is None

    @pytest.mark.asyncio
    async def test_delete_file(self, store, db):
        await store.upsert_file("memory/test.md", "memory", "abc", 100.0, 10)
        await store.commit()
        await store.delete_file("memory/test.md")
        await store.commit()
        assert await store.get_file("memory/test.md") is None

    @pytest.mark.asyncio
    async def test_list_files_empty(self, store):
        files = await store.list_files()
        # schema_version exists in meta, files table should be empty
        assert files == []

    @pytest.mark.asyncio
    async def test_list_files_by_source(self, store, db):
        await store.upsert_file("memory/a.md", "memory", "h1", 1.0, 10)
        await store.upsert_file("memory/researcher/b.md", "researcher", "h2", 2.0, 20)
        await store.commit()
        memory_files = await store.list_files(source="memory")
        assert len(memory_files) == 1
        assert memory_files[0]["source"] == "memory"


# ── SQLiteStore chunks tests ──────────────────────────────────────────────────


class TestSQLiteStoreChunks:
    @pytest.mark.asyncio
    async def test_upsert_and_get_chunk(self, store, db):
        embedding = [0.1, 0.2, 0.3]
        await store.upsert_chunk(
            id_="chunk1",
            path="memory/test.md",
            source="memory",
            start_line=1,
            end_line=5,
            hash_="chunkhash",
            model="text-embedding-3-small",
            text="Hello world",
            embedding=embedding,
        )
        await store.commit()

        record = await store.get_chunk("chunk1")
        assert record is not None
        assert record["text"] == "Hello world"
        assert record["embedding"] == embedding
        assert record["start_line"] == 1
        assert record["end_line"] == 5

    @pytest.mark.asyncio
    async def test_get_nonexistent_chunk(self, store):
        assert await store.get_chunk("does_not_exist") is None

    @pytest.mark.asyncio
    async def test_chunk_without_embedding(self, store, db):
        await store.upsert_chunk(
            id_="c2",
            path="memory/test.md",
            source="memory",
            start_line=1,
            end_line=2,
            hash_="h",
            model="model",
            text="text",
            embedding=None,
        )
        await store.commit()
        record = await store.get_chunk("c2")
        assert record["embedding"] is None

    @pytest.mark.asyncio
    async def test_delete_chunks_by_path(self, store, db):
        await store.upsert_chunk(
            id_="c1",
            path="memory/a.md",
            source="memory",
            start_line=1,
            end_line=2,
            hash_="h1",
            model="m",
            text="t1",
            embedding=None,
        )
        await store.upsert_chunk(
            id_="c2",
            path="memory/a.md",
            source="memory",
            start_line=3,
            end_line=4,
            hash_="h2",
            model="m",
            text="t2",
            embedding=None,
        )
        await store.commit()
        deleted = await store.delete_chunks_by_path("memory/a.md")
        await store.commit()
        assert deleted == 2
        assert await store.get_chunk("c1") is None

    @pytest.mark.asyncio
    async def test_count_chunks(self, store, db):
        assert await store.count_chunks() == 0
        await store.upsert_chunk(
            id_="c1",
            path="p",
            source="memory",
            start_line=1,
            end_line=2,
            hash_="h",
            model="m",
            text="t",
            embedding=None,
        )
        await store.commit()
        assert await store.count_chunks() == 1


# ── SQLiteStore FTS tests ─────────────────────────────────────────────────────


class TestSQLiteStoreFTS:
    async def test_lookup_survives_reopen_rollback_and_other_writers(self, tmp_path):
        database = tmp_path / "index.sqlite"
        async with aiosqlite.connect(database) as first, aiosqlite.connect(database) as second:
            await ensure_schema(first)
            await ensure_schema(second)
            a, b = SQLiteStore(first), SQLiteStore(second)
            await a.upsert_fts("original", "same", "a.md", "memory", 1, 1)
            await a.commit()
            await b.upsert_fts("other writer", "same", "b.md", "memory", 1, 1)
            await b.commit()
            await a.upsert_fts("rolled back", "same", "c.md", "memory", 1, 1)
            await a.rollback()
            rows = await (await first.execute("SELECT text,path FROM chunks_fts")).fetchall()
            assert rows == [("other writer", "b.md")]
        async with aiosqlite.connect(database) as reopened:
            await ensure_schema(reopened)
            store = SQLiteStore(reopened)
            await store.delete_fts_by_path("b.md")
            await store.commit()
            assert await (await reopened.execute("SELECT * FROM chunks_fts")).fetchall() == []
            assert (
                await (await reopened.execute("SELECT * FROM memweave_fts_lookup")).fetchall() == []
            )

    async def test_fts_metadata_lookups_use_btree_indexes(self, db):
        for column in ("id", "path"):
            rows = await (
                await db.execute(
                    f"EXPLAIN QUERY PLAN SELECT fts_rowid FROM memweave_fts_lookup WHERE {column} = ?",
                    ["needle"],
                )
            ).fetchall()
            assert any("USING COVERING INDEX" in row[3] for row in rows)

    @pytest.mark.asyncio
    async def test_upsert_and_search_fts(self, store, db):
        await store.upsert_fts(
            text="PostgreSQL chosen for JSONB support",
            chunk_id="c1",
            path="memory/test.md",
            source="memory",
            start_line=1,
            end_line=2,
        )
        await store.commit()

        cursor = await db.execute("SELECT id FROM chunks_fts WHERE chunks_fts MATCH 'PostgreSQL'")
        rows = await cursor.fetchall()
        assert any(row[0] == "c1" for row in rows)

    @pytest.mark.asyncio
    async def test_delete_fts_by_path(self, store, db):
        await store.upsert_fts("text", "c1", "memory/a.md", "memory", 1, 2)
        await store.commit()
        await store.delete_fts_by_path("memory/a.md")
        await store.commit()
        cursor = await db.execute("SELECT id FROM chunks_fts WHERE path = 'memory/a.md'")
        assert await cursor.fetchone() is None


# ── SQLiteStore embedding cache tests ────────────────────────────────────────


class TestSQLiteStoreEmbeddingCache:
    @pytest.mark.asyncio
    async def test_cache_miss(self, store):
        result = await store.get_embedding("openai", "model", "key", "missing_hash")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit(self, store, db):
        vec = [0.1, 0.2, 0.9]
        await store.upsert_embedding("openai", "model", "key", "hash1", vec, 3)
        await store.commit()
        cached = await store.get_embedding("openai", "model", "key", "hash1")
        assert cached == vec

    @pytest.mark.asyncio
    async def test_bulk_get_embeddings(self, store, db):
        await store.upsert_embedding("openai", "model", "key", "h1", [0.1], 1)
        await store.upsert_embedding("openai", "model", "key", "h2", [0.2], 1)
        await store.commit()
        result = await store.get_embeddings_bulk("openai", "model", "key", ["h1", "h2", "h3"])
        assert result["h1"] == [0.1]
        assert result["h2"] == [0.2]
        assert "h3" not in result  # cache miss

    @pytest.mark.asyncio
    async def test_count_cache_entries(self, store, db):
        assert await store.count_cache_entries() == 0
        await store.upsert_embedding("p", "m", "k", "h", [0.1], 1)
        await store.commit()
        assert await store.count_cache_entries() == 1

    @pytest.mark.asyncio
    async def test_clear_cache(self, store, db):
        await store.upsert_embedding("p", "m", "k", "h", [0.1], 1)
        await store.commit()
        count = await store.clear_cache()
        await store.commit()
        assert count == 1
        assert await store.count_cache_entries() == 0


# ── storage/files.py tests ───────────────────────────────────────────────────


class TestListMemoryFiles:
    def test_finds_md_files(self, tmp_path):
        memory = tmp_path / "memory"
        memory.mkdir()
        (memory / "2026-03-21.md").write_text("content")
        (memory / "MEMORY.md").write_text("bootstrap")

        files = list_memory_files(tmp_path)
        names = [f.name for f in files]
        assert "2026-03-21.md" in names
        assert "MEMORY.md" in names

    def test_ignores_non_md(self, tmp_path):
        memory = tmp_path / "memory"
        memory.mkdir()
        (memory / "notes.txt").write_text("not md")
        files = list_memory_files(tmp_path)
        assert all(f.suffix == ".md" for f in files)

    def test_recurses_into_subdirs(self, tmp_path):
        memory = tmp_path / "memory"
        (memory / "researcher").mkdir(parents=True)
        (memory / "researcher" / "log.md").write_text("research")
        files = list_memory_files(tmp_path)
        assert any("researcher" in f.parts for f in files)

    def test_empty_memory_dir(self, tmp_path):
        (tmp_path / "memory").mkdir()
        assert list_memory_files(tmp_path) == []

    def test_no_memory_dir(self, tmp_path):
        assert list_memory_files(tmp_path) == []

    def test_sorted_output(self, tmp_path):
        memory = tmp_path / "memory"
        memory.mkdir()
        (memory / "c.md").write_text("")
        (memory / "a.md").write_text("")
        (memory / "b.md").write_text("")
        files = list_memory_files(tmp_path)
        names = [f.name for f in files]
        assert names == sorted(names)


class TestGetSourceFromPath:
    def test_directly_under_memory(self, tmp_path):
        path = tmp_path / "memory" / "2026-03-21.md"
        assert get_source_from_path(path, tmp_path) == "memory"

    def test_subdirectory(self, tmp_path):
        path = tmp_path / "memory" / "researcher" / "log.md"
        assert get_source_from_path(path, tmp_path) == "researcher"

    def test_nested_subdirectory(self, tmp_path):
        path = tmp_path / "memory" / "sessions" / "deep" / "file.md"
        assert get_source_from_path(path, tmp_path) == "sessions"

    def test_outside_workspace(self, tmp_path):
        path = Path("/other/dir/file.md")
        assert get_source_from_path(path, tmp_path) == "external"


class TestIsEvergreen:
    def test_memory_md_is_evergreen(self, tmp_path):
        path = tmp_path / "memory" / "MEMORY.md"
        assert is_evergreen(path, ["MEMORY.md", "memory.md"]) is True

    def test_dated_file_is_not_evergreen(self, tmp_path):
        path = tmp_path / "memory" / "2026-03-21.md"
        assert is_evergreen(path, ["MEMORY.md"]) is False

    def test_non_dated_non_pattern_is_evergreen(self, tmp_path):
        # "decisions.md" is not dated (no YYYY-MM-DD.md pattern) → evergreen
        path = tmp_path / "memory" / "decisions.md"
        assert is_evergreen(path, ["MEMORY.md"]) is True


class TestIsMemoryPath:
    def test_inside_memory(self, tmp_path):
        path = tmp_path / "memory" / "test.md"
        assert is_memory_path(path, tmp_path) is True

    def test_outside_memory(self, tmp_path):
        path = tmp_path / "other" / "test.md"
        assert is_memory_path(path, tmp_path) is False


class TestRelativePath:
    def test_relative_to_workspace(self, tmp_path):
        path = tmp_path / "memory" / "2026-03-21.md"
        result = relative_path(path, tmp_path)
        assert result == "memory/2026-03-21.md"

    def test_outside_workspace_returns_absolute(self):
        result = relative_path(Path("/other/path.md"), Path("/workspace"))
        assert result == "/other/path.md"
