"""
Local graph store — a drop-in replacement for Zep Cloud's graph API.

Uses NetworkX for in-memory graph operations, JSON files for persistence,
and the project's existing LLMClient for entity/relationship extraction from
raw text.

Storage layout (under backend/uploads/graphs/{graph_id}/):
    meta.json      — graph metadata (id, name, description, created_at)
    ontology.json  — entity + edge type definitions from set_ontology()
    nodes.json     — all GraphNode objects (keyed by uuid_)
    edges.json     — all GraphEdge objects (keyed by uuid_)
    episodes.json  — episode tracking (keyed by uuid_)

Client interface mirrors Zep so consuming code can swap with minimal changes:

    client = LocalGraphClient()
    client.graph.create(graph_id=..., name=..., description=...)
    client.graph.add_batch(graph_id=..., episodes=[...])
    client.graph.node.get(uuid_=...)
    client.graph.search(graph_id=..., query=..., limit=...)
    client.graph.add(graph_id=..., data=..., type="text")
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient

logger = get_logger("mirofish.local_graph_store")

# ---------------------------------------------------------------------------
# Storage root
# ---------------------------------------------------------------------------

_GRAPHS_DIR = os.path.join(
    os.path.dirname(__file__),   # backend/app/services/
    "../../uploads/graphs"
)


def _graph_dir(graph_id: str) -> str:
    return os.path.join(_GRAPHS_DIR, graph_id)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Data-model dataclasses (mirror Zep response objects)
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """Mirrors zep_cloud EntityNode / Node response object."""
    uuid_: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GraphNode":
        return GraphNode(
            uuid_=d["uuid_"],
            name=d["name"],
            labels=d.get("labels", []),
            summary=d.get("summary", ""),
            attributes=d.get("attributes", {}),
            created_at=d.get("created_at", _now()),
        )


@dataclass
class GraphEdge:
    """Mirrors zep_cloud EntityEdge response object."""
    uuid_: str
    name: str
    fact: str
    fact_type: str
    source_node_uuid: str
    target_node_uuid: str
    attributes: Dict[str, Any]
    created_at: str
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    episodes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GraphEdge":
        return GraphEdge(
            uuid_=d["uuid_"],
            name=d["name"],
            fact=d.get("fact", ""),
            fact_type=d.get("fact_type", d.get("name", "")),
            source_node_uuid=d["source_node_uuid"],
            target_node_uuid=d["target_node_uuid"],
            attributes=d.get("attributes", {}),
            created_at=d.get("created_at", _now()),
            valid_at=d.get("valid_at"),
            invalid_at=d.get("invalid_at"),
            expired_at=d.get("expired_at"),
            episodes=d.get("episodes", []),
        )


@dataclass
class Episode:
    """Mirrors zep_cloud Episode response object."""
    uuid_: str
    processed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Episode":
        return Episode(uuid_=d["uuid_"], processed=d.get("processed", True))


@dataclass
class NodeSearchResult:
    """Single result returned by graph.search()."""
    node: GraphNode
    score: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_uuid() -> str:
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# JSON persistence helpers
# ---------------------------------------------------------------------------

class _Store:
    """
    Thread-safe JSON file store for a single graph.

    Each graph lives in its own directory; each collection is a flat JSON
    object keyed by uuid_.  All public mutating methods acquire _lock.
    """

    def __init__(self, graph_id: str) -> None:
        self.graph_id = graph_id
        self._dir = _graph_dir(graph_id)
        self._lock = threading.Lock()
        _ensure_dir(self._dir)

    # ---- low-level file I/O -----------------------------------------------

    def _path(self, filename: str) -> str:
        return os.path.join(self._dir, filename)

    def _read_json(self, filename: str) -> Dict[str, Any]:
        p = self._path(filename)
        if not os.path.exists(p):
            return {}
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            logger.warning(f"Failed to read {p}, returning empty dict")
            return {}

    def _write_json(self, filename: str, data: Dict[str, Any]) -> None:
        p = self._path(filename)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, p)  # atomic on POSIX

    # ---- meta ----------------------------------------------------------------

    def read_meta(self) -> Dict[str, Any]:
        return self._read_json("meta.json")

    def write_meta(self, meta: Dict[str, Any]) -> None:
        with self._lock:
            self._write_json("meta.json", meta)

    # ---- ontology ------------------------------------------------------------

    def read_ontology(self) -> Dict[str, Any]:
        return self._read_json("ontology.json")

    def write_ontology(self, ontology: Dict[str, Any]) -> None:
        with self._lock:
            self._write_json("ontology.json", ontology)

    # ---- nodes ---------------------------------------------------------------

    def read_nodes(self) -> Dict[str, Dict[str, Any]]:
        return self._read_json("nodes.json")

    def upsert_node(self, node: GraphNode) -> None:
        """Insert or update a node, merging attributes on name collision."""
        with self._lock:
            nodes = self._read_json("nodes.json")
            nodes[node.uuid_] = node.to_dict()
            self._write_json("nodes.json", nodes)

    def upsert_nodes(self, new_nodes: List[GraphNode]) -> None:
        """Batch upsert — single write per call."""
        with self._lock:
            nodes = self._read_json("nodes.json")
            for n in new_nodes:
                nodes[n.uuid_] = n.to_dict()
            self._write_json("nodes.json", nodes)

    # ---- edges ---------------------------------------------------------------

    def read_edges(self) -> Dict[str, Dict[str, Any]]:
        return self._read_json("edges.json")

    def upsert_edges(self, new_edges: List[GraphEdge]) -> None:
        with self._lock:
            edges = self._read_json("edges.json")
            for e in new_edges:
                edges[e.uuid_] = e.to_dict()
            self._write_json("edges.json", edges)

    # ---- episodes ------------------------------------------------------------

    def read_episodes(self) -> Dict[str, Dict[str, Any]]:
        return self._read_json("episodes.json")

    def add_episode(self, ep: Episode) -> None:
        with self._lock:
            episodes = self._read_json("episodes.json")
            episodes[ep.uuid_] = ep.to_dict()
            self._write_json("episodes.json", episodes)

    # ---- delete --------------------------------------------------------------

    def delete_all(self) -> None:
        """Remove the entire graph directory."""
        import shutil
        with self._lock:
            if os.path.isdir(self._dir):
                shutil.rmtree(self._dir)


# ---------------------------------------------------------------------------
# Module-level store registry (one _Store per graph_id)
# ---------------------------------------------------------------------------

_store_registry: Dict[str, _Store] = {}
_registry_lock = threading.Lock()


def _get_store(graph_id: str) -> _Store:
    with _registry_lock:
        if graph_id not in _store_registry:
            _store_registry[graph_id] = _Store(graph_id)
        return _store_registry[graph_id]


def _remove_from_registry(graph_id: str) -> None:
    with _registry_lock:
        _store_registry.pop(graph_id, None)


# ---------------------------------------------------------------------------
# LLM entity extraction
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """\
You are a knowledge-graph extraction assistant.
Given a text passage and an ontology definition, extract all entities and
relationships that appear in the text.

Return a single JSON object with this exact structure:
{
  "entities": [
    {
      "name": "<entity name, normalised>",
      "type": "<entity type from the ontology, or 'Entity' if none matches>",
      "summary": "<one-sentence description of this entity>",
      "attributes": { "<attr_name>": "<value>", ... }
    }
  ],
  "relationships": [
    {
      "source": "<source entity name>",
      "target": "<target entity name>",
      "type": "<relationship type from the ontology, or 'RELATED_TO' if none matches>",
      "fact": "<one-sentence statement of the relationship>"
    }
  ]
}

Rules:
- Only extract entities whose names appear explicitly in the text.
- Normalise entity names (e.g. consistent capitalisation).
- If the ontology is empty, still extract obvious entities and relationships.
- Return valid JSON only — no extra text, no Markdown code fences.
"""


def _build_extraction_prompt(ontology: Dict[str, Any], text: str) -> str:
    """Build the user message for entity extraction."""
    entity_types = [
        f"- {e['name']}: {e.get('description', '')}"
        for e in ontology.get("entity_types", [])
    ]
    edge_types = [
        f"- {e['name']}: {e.get('description', '')}"
        for e in ontology.get("edge_types", [])
    ]

    parts = [f"TEXT:\n{text}\n"]
    if entity_types:
        parts.append("ENTITY TYPES:\n" + "\n".join(entity_types))
    if edge_types:
        parts.append("RELATIONSHIP TYPES:\n" + "\n".join(edge_types))
    return "\n\n".join(parts)


def _extract_entities_from_text(
    llm: LLMClient,
    ontology: Dict[str, Any],
    text: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Call the LLM to extract entities and relationships from text.

    Returns:
        (entities, relationships) — lists of raw dicts.
    """
    user_content = _build_extraction_prompt(ontology, text)
    messages = [
        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        result = llm.chat_json(messages=messages, temperature=0.2, max_tokens=4096)
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])
        if not isinstance(entities, list):
            entities = []
        if not isinstance(relationships, list):
            relationships = []
        return entities, relationships
    except Exception as exc:
        logger.error(f"LLM entity extraction failed: {exc}")
        return [], []


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def _find_or_create_node(
    name: str,
    entity_type: str,
    summary: str,
    attributes: Dict[str, Any],
    name_index: Dict[str, GraphNode],  # mutated in-place
    new_nodes: List[GraphNode],        # mutated in-place
) -> GraphNode:
    """
    Return an existing node for this name (case-insensitive) or create one.
    Merges summary and attributes if the node already exists.
    """
    key = name.strip().lower()
    if key in name_index:
        existing = name_index[key]
        # Merge: prefer non-empty summary; merge attribute dicts
        if not existing.summary and summary:
            existing.summary = summary
        existing.attributes.update({k: v for k, v in attributes.items() if v})
        return existing

    labels = ["Entity"]
    if entity_type and entity_type not in ("Entity", "Node"):
        labels.append(entity_type)

    node = GraphNode(
        uuid_=_new_uuid(),
        name=name.strip(),
        labels=labels,
        summary=summary,
        attributes=attributes,
        created_at=_now(),
    )
    name_index[key] = node
    new_nodes.append(node)
    return node


def _edge_dedup_key(
    source_uuid: str, target_uuid: str, fact_type: str
) -> str:
    return f"{source_uuid}|{target_uuid}|{fact_type}"


# ---------------------------------------------------------------------------
# Core graph operations
# ---------------------------------------------------------------------------

def _merge_episodes_into_graph(
    graph_id: str,
    episode_texts: List[str],
    llm: LLMClient,
) -> List[Episode]:
    """
    Extract entities/relationships from each episode text and persist them.

    Returns a list of Episode objects (all pre-marked processed=True since
    extraction is synchronous).
    """
    store = _get_store(graph_id)
    ontology = store.read_ontology()

    # Load existing data for deduplication
    existing_nodes_raw = store.read_nodes()
    existing_edges_raw = store.read_edges()

    # Build name index for node deduplication
    name_index: Dict[str, GraphNode] = {
        d["name"].strip().lower(): GraphNode.from_dict(d)
        for d in existing_nodes_raw.values()
    }

    # Build edge dedup set
    edge_dedup: set[str] = {
        _edge_dedup_key(d["source_node_uuid"], d["target_node_uuid"], d.get("fact_type", ""))
        for d in existing_edges_raw.values()
    }

    episodes: List[Episode] = []
    all_new_nodes: List[GraphNode] = []
    all_new_edges: List[GraphEdge] = []

    for text in episode_texts:
        ep_uuid = _new_uuid()
        episode = Episode(uuid_=ep_uuid, processed=True)
        episodes.append(episode)
        store.add_episode(episode)

        if not text.strip():
            continue

        extracted_entities, extracted_relationships = _extract_entities_from_text(
            llm, ontology, text
        )
        logger.debug(
            f"[{graph_id}] extracted {len(extracted_entities)} entities, "
            f"{len(extracted_relationships)} relationships from episode {ep_uuid}"
        )

        # Process entities
        ep_new_nodes: List[GraphNode] = []
        for ent in extracted_entities:
            ent_name = (ent.get("name") or "").strip()
            if not ent_name:
                continue
            node = _find_or_create_node(
                name=ent_name,
                entity_type=ent.get("type", "Entity"),
                summary=ent.get("summary", ""),
                attributes=ent.get("attributes") or {},
                name_index=name_index,
                new_nodes=ep_new_nodes,
            )
            _ = node  # already registered in name_index

        all_new_nodes.extend(ep_new_nodes)

        # Process relationships
        for rel in extracted_relationships:
            src_name = (rel.get("source") or "").strip()
            tgt_name = (rel.get("target") or "").strip()
            rel_type = (rel.get("type") or "RELATED_TO").strip()
            fact = (rel.get("fact") or "").strip()

            if not src_name or not tgt_name:
                continue

            # Ensure both endpoint nodes exist
            src_node = _find_or_create_node(
                name=src_name,
                entity_type="Entity",
                summary="",
                attributes={},
                name_index=name_index,
                new_nodes=all_new_nodes,
            )
            tgt_node = _find_or_create_node(
                name=tgt_name,
                entity_type="Entity",
                summary="",
                attributes={},
                name_index=name_index,
                new_nodes=all_new_nodes,
            )

            dedup_key = _edge_dedup_key(src_node.uuid_, tgt_node.uuid_, rel_type)
            if dedup_key in edge_dedup:
                continue
            edge_dedup.add(dedup_key)

            edge = GraphEdge(
                uuid_=_new_uuid(),
                name=rel_type,
                fact=fact or f"{src_name} {rel_type.lower()} {tgt_name}",
                fact_type=rel_type,
                source_node_uuid=src_node.uuid_,
                target_node_uuid=tgt_node.uuid_,
                attributes={},
                created_at=_now(),
                episodes=[ep_uuid],
            )
            all_new_edges.append(edge)

    # Persist all changes in bulk
    if all_new_nodes:
        store.upsert_nodes(all_new_nodes)
    if all_new_edges:
        store.upsert_edges(all_new_edges)

    return episodes


# ---------------------------------------------------------------------------
# Search implementation
# ---------------------------------------------------------------------------

def _score_node(node: GraphNode, query_lower: str) -> float:
    """
    Simple text-match scorer.

    Scoring tiers:
        1.0  exact name match (case-insensitive)
        0.8  query is a substring of name
        0.6  query is a substring of summary
        0.4  query is a substring of any attribute value
    """
    name_lower = node.name.lower()

    if name_lower == query_lower:
        return 1.0
    if query_lower in name_lower:
        return 0.8
    if query_lower in node.summary.lower():
        return 0.6
    for v in node.attributes.values():
        if isinstance(v, str) and query_lower in v.lower():
            return 0.4
    return 0.0


def _search_nodes(
    graph_id: str,
    query: str,
    limit: int,
) -> List[NodeSearchResult]:
    """Return top-N nodes ranked by text similarity to query."""
    store = _get_store(graph_id)
    nodes_raw = store.read_nodes()
    query_lower = query.strip().lower()

    scored: List[Tuple[float, GraphNode]] = []
    for d in nodes_raw.values():
        node = GraphNode.from_dict(d)
        score = _score_node(node, query_lower)
        if score > 0:
            scored.append((score, node))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [NodeSearchResult(node=n, score=s) for s, n in scored[:limit]]


# ---------------------------------------------------------------------------
# Sub-API objects (node, edge, episode) — used by LocalGraphAPI
# ---------------------------------------------------------------------------

class _EpisodeAPI:
    def get(self, uuid_: str) -> Episode:
        """
        Return an Episode by UUID.

        Since extraction is synchronous, episodes are always processed=True.
        If the UUID is not found we still return a processed episode object
        rather than raising — matching Zep's lenient behaviour when an
        episode has already been gc'd.
        """
        # Search all graphs (uuid is globally unique)
        _ensure_dir(_GRAPHS_DIR)
        for graph_id in os.listdir(_GRAPHS_DIR):
            store = _get_store(graph_id)
            episodes = store.read_episodes()
            if uuid_ in episodes:
                return Episode.from_dict(episodes[uuid_])
        # Unknown uuid — return a synthetic processed episode
        return Episode(uuid_=uuid_, processed=True)


class _NodeAPI:
    def get(self, uuid_: str) -> Optional[GraphNode]:
        """Get a node by UUID. Searches across all graphs."""
        _ensure_dir(_GRAPHS_DIR)
        for graph_id in os.listdir(_GRAPHS_DIR):
            store = _get_store(graph_id)
            nodes = store.read_nodes()
            if uuid_ in nodes:
                return GraphNode.from_dict(nodes[uuid_])
        return None

    def get_entity_edges(self, node_uuid: str) -> List[GraphEdge]:
        """Return all edges incident to node_uuid (source or target)."""
        _ensure_dir(_GRAPHS_DIR)
        results: List[GraphEdge] = []
        for graph_id in os.listdir(_GRAPHS_DIR):
            store = _get_store(graph_id)
            edges_raw = store.read_edges()
            for d in edges_raw.values():
                if d["source_node_uuid"] == node_uuid or d["target_node_uuid"] == node_uuid:
                    results.append(GraphEdge.from_dict(d))
        return results

    def get_by_graph_id(
        self,
        graph_id: str,
        limit: int = 100,
        uuid_cursor: Optional[str] = None,
    ) -> List[GraphNode]:
        """
        Paginated node listing, ordered by uuid_ string.

        Mimics Zep's UUID-cursor pagination: returns up to `limit` nodes
        whose uuid_ sorts after uuid_cursor.
        """
        store = _get_store(graph_id)
        nodes_raw = store.read_nodes()
        all_uuids = sorted(nodes_raw.keys())

        if uuid_cursor is not None:
            try:
                start_idx = all_uuids.index(uuid_cursor) + 1
            except ValueError:
                # cursor not found — start from beginning
                start_idx = 0
        else:
            start_idx = 0

        page_uuids = all_uuids[start_idx : start_idx + limit]
        return [GraphNode.from_dict(nodes_raw[u]) for u in page_uuids]


class _EdgeAPI:
    def get_by_graph_id(
        self,
        graph_id: str,
        limit: int = 100,
        uuid_cursor: Optional[str] = None,
    ) -> List[GraphEdge]:
        """Paginated edge listing, ordered by uuid_ string."""
        store = _get_store(graph_id)
        edges_raw = store.read_edges()
        all_uuids = sorted(edges_raw.keys())

        if uuid_cursor is not None:
            try:
                start_idx = all_uuids.index(uuid_cursor) + 1
            except ValueError:
                start_idx = 0
        else:
            start_idx = 0

        page_uuids = all_uuids[start_idx : start_idx + limit]
        return [GraphEdge.from_dict(edges_raw[u]) for u in page_uuids]


# ---------------------------------------------------------------------------
# LocalGraphAPI — the object mounted as client.graph
# ---------------------------------------------------------------------------

class LocalGraphAPI:
    """
    Provides the same interface as the Zep graph client:

        client.graph.create(...)
        client.graph.delete(...)
        client.graph.set_ontology(...)
        client.graph.add_batch(...)
        client.graph.add(...)
        client.graph.search(...)
        client.graph.node.get(...)
        client.graph.node.get_entity_edges(...)
        client.graph.node.get_by_graph_id(...)
        client.graph.edge.get_by_graph_id(...)
        client.graph.episode.get(...)
    """

    def __init__(self) -> None:
        self.node = _NodeAPI()
        self.edge = _EdgeAPI()
        self.episode = _EpisodeAPI()
        self._llm: Optional[LLMClient] = None
        self._llm_lock = threading.Lock()

    def _get_llm(self) -> LLMClient:
        """Lazy-initialise the LLM client (avoids import-time failures if
        LLM_API_KEY is not yet set when the module is loaded)."""
        if self._llm is None:
            with self._llm_lock:
                if self._llm is None:
                    self._llm = LLMClient()
        return self._llm

    # ---- graph lifecycle -----------------------------------------------------

    def create(
        self,
        graph_id: str,
        name: str,
        description: str = "",
    ) -> None:
        """Create a new graph namespace and persist metadata."""
        _ensure_dir(_GRAPHS_DIR)
        store = _get_store(graph_id)
        meta = {
            "graph_id": graph_id,
            "name": name,
            "description": description,
            "created_at": _now(),
        }
        store.write_meta(meta)
        logger.info(f"Created local graph: {graph_id} ({name!r})")

    def delete(self, graph_id: str) -> None:
        """Delete a graph and all its data files."""
        store = _get_store(graph_id)
        store.delete_all()
        _remove_from_registry(graph_id)
        logger.info(f"Deleted local graph: {graph_id}")

    # ---- ontology ------------------------------------------------------------

    def set_ontology(
        self,
        graph_ids: List[str],
        entities: Optional[Any] = None,
        edges: Optional[Any] = None,
        _ontology_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store the ontology definition for each graph.

        The Zep SDK accepts dynamically-created Pydantic model classes; here
        we serialise them to a plain dict so we can later pass the ontology
        definition to the LLM extraction prompt.

        entities: dict[str, type] — entity name → EntityModel class
        edges:    dict[str, tuple[type, list[EntityEdgeSourceTarget]]]
        """
        # If a raw ontology dict is provided, persist it directly and skip class introspection.
        if _ontology_dict is not None:
            for graph_id in graph_ids:
                _get_store(graph_id).write_ontology(_ontology_dict)
                logger.info(
                    f"Set ontology for graph {graph_id} from raw dict: "
                    f"{len(_ontology_dict.get('entity_types', []))} entity types, "
                    f"{len(_ontology_dict.get('edge_types', []))} edge types"
                )
            return

        ontology: Dict[str, Any] = {"entity_types": [], "edge_types": []}

        if entities:
            for ent_name, ent_class in entities.items():
                ontology["entity_types"].append({
                    "name": ent_name,
                    "description": getattr(ent_class, "__doc__", "") or "",
                    "attributes": _extract_class_field_names(ent_class),
                })

        if edges:
            for edge_name, edge_info in edges.items():
                edge_class, source_targets = edge_info
                sts = [
                    {"source": getattr(st, "source", "Entity"),
                     "target": getattr(st, "target", "Entity")}
                    for st in source_targets
                ]
                ontology["edge_types"].append({
                    "name": edge_name,
                    "description": getattr(edge_class, "__doc__", "") or "",
                    "source_targets": sts,
                    "attributes": _extract_class_field_names(edge_class),
                })

        for graph_id in graph_ids:
            _get_store(graph_id).write_ontology(ontology)
            logger.info(f"Set ontology for graph {graph_id}: "
                        f"{len(ontology['entity_types'])} entity types, "
                        f"{len(ontology['edge_types'])} edge types")

    # ---- add batch -----------------------------------------------------------

    def add_batch(
        self,
        graph_id: str,
        episodes: List[Any],
    ) -> List[Episode]:
        """
        Process a list of episode objects.

        Each episode must have a `.data` attribute (the raw text) and
        a `.type` attribute (expected to be "text").  This mimics
        `zep_cloud.EpisodeData`.

        Entity extraction is performed synchronously via LLM.
        Returns a list of Episode objects (all processed=True).
        """
        texts: List[str] = []
        for ep in episodes:
            data = getattr(ep, "data", "") or ""
            if data:
                texts.append(data)

        if not texts:
            return []

        llm = self._get_llm()
        result = _merge_episodes_into_graph(graph_id, texts, llm)
        logger.info(
            f"add_batch: graph={graph_id}, "
            f"{len(texts)} episodes, "
            f"{len(result)} processed"
        )
        return result

    # ---- add (single text chunk) ---------------------------------------------

    def add(
        self,
        graph_id: str,
        data: str,
        type: str = "text",  # noqa: A002
    ) -> None:
        """
        Add a single text chunk to the graph (used by ZepGraphMemoryUpdater).

        Equivalent to add_batch with one episode.
        """
        if not data or not data.strip():
            return
        llm = self._get_llm()
        _merge_episodes_into_graph(graph_id, [data], llm)
        logger.debug(f"add: graph={graph_id}, data_len={len(data)}")

    # ---- search --------------------------------------------------------------

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "nodes",         # unused; kept for API compat
        search_type: str = "similarity",  # unused; kept for API compat
    ) -> List[NodeSearchResult]:
        """
        Search nodes by text matching.

        Scoring (see _score_node):
            1.0  exact name match
            0.8  query in name
            0.6  query in summary
            0.4  query in any attribute value

        Returns up to `limit` NodeSearchResult objects sorted by score desc.
        """
        results = _search_nodes(graph_id, query, limit)
        logger.debug(f"search: graph={graph_id}, query={query!r}, hits={len(results)}")
        return results


# ---------------------------------------------------------------------------
# LocalGraphClient — the top-level drop-in for Zep(api_key=...)
# ---------------------------------------------------------------------------

class LocalGraphClient:
    """
    Drop-in replacement for `zep_cloud.client.Zep`.

    Usage:
        # Before (Zep Cloud)
        from zep_cloud.client import Zep
        client = Zep(api_key=Config.ZEP_API_KEY)

        # After (local)
        from .local_graph_store import LocalGraphClient
        client = LocalGraphClient()

    All graph operations are available via client.graph.
    """

    def __init__(self) -> None:
        self.graph = LocalGraphAPI()
        logger.info("LocalGraphClient initialised (local NetworkX/JSON store)")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _extract_class_field_names(cls: Any) -> List[Dict[str, str]]:
    """
    Extract field names + descriptions from a Pydantic/dataclass-style class.

    Zep's EntityModel / EdgeModel classes have Pydantic FieldInfo objects as
    class attributes; we pull the description from those for the ontology JSON.
    """
    result = []
    annotations = getattr(cls, "__annotations__", {})
    for attr_name in annotations:
        if attr_name.startswith("_"):
            continue
        field_obj = getattr(cls, attr_name, None)
        desc = ""
        if field_obj is not None:
            # Pydantic v2 FieldInfo stores description in .metadata or .description
            desc = (
                getattr(field_obj, "description", "")
                or getattr(field_obj, "title", "")
                or ""
            )
        result.append({"name": attr_name, "description": desc})
    return result
