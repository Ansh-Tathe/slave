"""
services/analytics/behaviour/gathering.py
==========================================
IBVAP P5 -- Crowd gathering detector.

Emits GATHERING when >= min_count persons are clustered within
proximity_px of each other for >= duration_s seconds.

Algorithm (simple O(n^2) proximity graph)
------------------------------------------
1. Build an adjacency list: two tracks are "neighbours" if their
   centroid distance <= proximity_px.
2. Find connected components (BFS) — each component is a cluster.
3. Any cluster with >= min_count members that has persisted for
   >= duration_s seconds triggers a GATHERING event.

Cluster identity is tracked by the frozenset of track_ids.  When a
cluster changes membership significantly (Jaccard similarity < 0.5),
the timer resets.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from loguru import logger

from services.detect_track.models import Track
from services.event_engine.models  import Event, EventType, Severity


@dataclass
class _ClusterState:
    first_seen:   float
    alerted:      bool = False
    alert_time:   float = 0.0


def _centroid_dist(a: Track, b: Track) -> float:
    ax, ay = a.center
    bx, by = b.center
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _connected_components(
    tracks: List[Track], proximity_px: float
) -> List[Set[int]]:
    """
    Find clusters of tracks within proximity_px of each other (BFS).
    Returns list of sets of track_ids.
    """
    if not tracks:
        return []

    idx    = {t.track_id: t for t in tracks}
    ids    = list(idx.keys())
    visited: Set[int] = set()
    components: List[Set[int]] = []

    for start in ids:
        if start in visited:
            continue
        cluster: Set[int] = set()
        queue   = [start]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            cluster.add(cur)
            for other in ids:
                if other not in visited:
                    if _centroid_dist(idx[cur], idx[other]) <= proximity_px:
                        queue.append(other)
        components.append(cluster)

    return components


def _jaccard(a: FrozenSet, b: FrozenSet) -> float:
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


class GatheringDetector:
    """
    Detects crowd gatherings from track clusters.

    Parameters
    ----------
    min_count       : minimum number of persons in cluster to alert
    proximity_px    : maximum centroid distance to consider two persons "together"
    duration_s      : cluster must persist for this many seconds before alert
    classes         : which class names count as crowd members (default: person)
    dedup_window_s  : suppress repeated GATHERING for same cluster
    severity        : event severity
    camera_id       : used in emitted events
    """

    def __init__(
        self,
        min_count:      int = 5,
        proximity_px:   float = 200.0,
        duration_s:     float = 10.0,
        classes:        Optional[List[str]] = None,
        dedup_window_s: float = 30.0,
        severity:       Severity = Severity.HIGH,
        camera_id:      str = "unknown",
    ) -> None:
        self._min_count   = min_count
        self._proximity   = proximity_px
        self._duration    = duration_s
        self._classes     = set(classes or ["person"])
        self._dedup_w     = dedup_window_s
        self._severity    = severity
        self._camera      = camera_id
        # cluster_key -> _ClusterState
        self._clusters: Dict[FrozenSet[int], _ClusterState] = {}

        logger.info(
            f"GatheringDetector ready — min_count={min_count}, "
            f"proximity={proximity_px}px, duration={duration_s}s"
        )

    def update(
        self,
        tracks:    List[Track],
        frame_id:  int,
        camera_id: Optional[str] = None,
    ) -> List[Event]:
        cam    = camera_id or self._camera
        now    = time.monotonic()
        events: List[Event] = []

        # Filter to relevant classes
        candidates = [t for t in tracks if t.class_name in self._classes]
        if len(candidates) < self._min_count:
            self._clusters.clear()
            return []

        # Find clusters
        components = _connected_components(candidates, self._proximity)
        large      = [c for c in components if len(c) >= self._min_count]

        # Match current clusters to known clusters (by Jaccard)
        new_clusters: Dict[FrozenSet[int], _ClusterState] = {}

        for cluster_set in large:
            key = frozenset(cluster_set)
            # Find best matching old cluster
            best_old: Optional[FrozenSet[int]] = None
            best_sim = 0.0
            for old_key in self._clusters:
                sim = _jaccard(key, old_key)
                if sim > best_sim:
                    best_sim = sim
                    best_old = old_key

            if best_old is not None and best_sim >= 0.5:
                # Continued cluster -- inherit state
                state = self._clusters[best_old]
            else:
                # New cluster -- start timer
                state = _ClusterState(first_seen=now)

            new_clusters[key] = state

            # Check duration
            age = now - state.first_seen
            if age >= self._duration:
                if (now - state.alert_time) >= self._dedup_w:
                    state.alerted    = True
                    state.alert_time = now

                    # Use the track with highest confidence as representative
                    tid_map    = {t.track_id: t for t in candidates}
                    rep_tracks = [tid_map[tid] for tid in cluster_set if tid in tid_map]
                    rep_track  = max(rep_tracks, key=lambda t: t.conf)

                    events.append(Event(
                        event_type = EventType.GATHERING,
                        severity   = self._severity,
                        camera_id  = cam,
                        track      = rep_track,
                        frame_id   = frame_id,
                        zone_id    = None,
                        metadata   = {
                            "cluster_size":  len(cluster_set),
                            "track_ids":     sorted(cluster_set),
                            "duration_s":    round(age, 1),
                            "proximity_px":  self._proximity,
                        },
                    ))
                    logger.info(
                        f"[{cam}] GATHERING  "
                        f"size={len(cluster_set)}  "
                        f"duration={age:.1f}s  "
                        f"track_ids={sorted(cluster_set)}"
                    )

        self._clusters = new_clusters
        return events

    def reset(self) -> None:
        self._clusters.clear()
