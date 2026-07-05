"""Core model implementation for TRUST-ST consensus clustering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


LabelCollection = Sequence[Sequence[object]]


@dataclass(frozen=True)
class TRUSTSTConfig:
    """Configuration for the reliability-guided consensus model."""

    cluster_reliability_theta: float = 0.50
    decorrelation_alpha: float = 0.15
    decorrelation_ridge: float = 0.25
    spatial_refine_k: int = 10
    confidence_quantile: float = 0.45
    local_entropy_quantile: float = 0.65
    majority_support: float = 0.60

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def encode_labels(labels: Sequence[object]) -> np.ndarray:
    """Encode arbitrary labels as consecutive integer cluster identifiers."""
    values = np.asarray(labels).astype(str)
    _, encoded = np.unique(values, return_inverse=True)
    return encoded.astype(np.int64)


def validate_inputs(
    base_partitions: LabelCollection,
    spatial_coordinates: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Validate base partitions and spatial coordinates."""
    if len(base_partitions) == 0:
        raise ValueError("base_partitions must contain at least one label vector.")

    partitions = [encode_labels(labels) for labels in base_partitions]
    n_spots = len(partitions[0])

    if any(len(labels) != n_spots for labels in partitions):
        raise ValueError("All base partitions must contain the same number of spots.")

    coordinates = np.asarray(spatial_coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] != n_spots:
        raise ValueError("spatial_coordinates must be an n_spots by d coordinate matrix.")

    return partitions, coordinates


def normalize_affinity(affinity: np.ndarray) -> np.ndarray:
    """Symmetrize, clip, scale, and diagonal-normalize an affinity matrix."""
    matrix = np.asarray(affinity, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)
    matrix = 0.5 * (matrix + matrix.T)
    matrix = np.clip(matrix, 0.0, None)

    maximum = float(matrix.max()) if matrix.size else 0.0
    if maximum > 0.0:
        matrix = matrix / maximum

    np.fill_diagonal(matrix, 1.0)
    return matrix


def plain_coassociation_matrix(base_partitions: LabelCollection) -> np.ndarray:
    """Construct the equal-voting plain co-association matrix."""
    partitions = [np.asarray(labels) for labels in base_partitions]
    n_methods = len(partitions)
    n_spots = len(partitions[0])

    affinity = np.zeros((n_spots, n_spots), dtype=np.float32)
    for labels in partitions:
        affinity += (labels[:, None] == labels[None, :]) / n_methods

    np.fill_diagonal(affinity, 1.0)
    return affinity


def method_decorrelation(
    base_partitions: LabelCollection,
    ridge: float = 0.25,
    alpha: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate method weights from an NMI-based redundancy matrix."""
    partitions = [np.asarray(labels) for labels in base_partitions]
    n_methods = len(partitions)
    similarity = np.eye(n_methods, dtype=float)

    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            nmi = normalized_mutual_info_score(partitions[i], partitions[j])
            similarity[i, j] = nmi
            similarity[j, i] = nmi

    raw_scores = np.linalg.solve(
        similarity + ridge * np.eye(n_methods, dtype=float),
        np.ones(n_methods, dtype=float),
    )
    lower_bound = 0.05 * float(np.mean(np.abs(raw_scores)))
    raw_scores = np.maximum(raw_scores, lower_bound)

    method_weights = raw_scores**alpha
    method_weights = method_weights / method_weights.sum()
    return method_weights, similarity, raw_scores


def cluster_reliability(
    base_partitions: LabelCollection,
    theta: float = 0.50,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
    """Estimate split-entropy-based reliability for each cluster unit."""
    partitions = [np.asarray(labels) for labels in base_partitions]
    n_methods = len(partitions)
    n_spots = len(partitions[0])
    spot_reliability = np.ones((n_methods, n_spots), dtype=float)
    unit_reliability: Dict[Tuple[int, int], float] = {}

    for method_idx, labels in enumerate(partitions):
        for cluster_id in np.unique(labels):
            support_set = labels == cluster_id
            normalized_entropies: List[float] = []

            for other_idx, other_labels in enumerate(partitions):
                if other_idx == method_idx:
                    continue

                _, counts = np.unique(other_labels[support_set], return_counts=True)
                probabilities = counts / counts.sum()
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
                normalizer = np.log(max(len(counts), 2))
                normalized_entropies.append(float(entropy / normalizer))

            uncertainty = float(np.mean(normalized_entropies)) if normalized_entropies else 0.0
            reliability = float(np.exp(-uncertainty / theta))
            spot_reliability[method_idx, support_set] = reliability
            unit_reliability[(method_idx, int(cluster_id))] = reliability

    return spot_reliability, unit_reliability


def reliability_weighted_affinity(
    base_partitions: LabelCollection,
    method_weights: Sequence[float],
    spot_reliability: np.ndarray,
) -> np.ndarray:
    """Build the reliability-weighted affinity matrix from pair-specific evidence."""
    partitions = [np.asarray(labels) for labels in base_partitions]
    n_spots = len(partitions[0])
    numerator = np.zeros((n_spots, n_spots), dtype=np.float32)
    denominator = np.zeros((n_spots, n_spots), dtype=np.float32)

    for method_idx, labels in enumerate(partitions):
        spot_strength = float(method_weights[method_idx]) * spot_reliability[method_idx]
        pair_weight = np.sqrt(spot_strength[:, None] * spot_strength[None, :])
        pair_weight = pair_weight.astype(np.float32)

        numerator += pair_weight * (labels[:, None] == labels[None, :])
        denominator += pair_weight

    affinity = numerator / np.maximum(denominator, 1e-12)
    np.fill_diagonal(affinity, 1.0)
    return normalize_affinity(affinity)


def adaptive_consensus_gate(
    spot_reliability: np.ndarray,
    plain_affinity: np.ndarray,
) -> np.ndarray:
    """Compute the pair-level gate for reliability-weighted correction."""
    n_methods, n_spots = spot_reliability.shape
    pair_mean = np.zeros((n_spots, n_spots), dtype=np.float32)
    pair_second_moment = np.zeros((n_spots, n_spots), dtype=np.float32)

    for method_idx in range(n_methods):
        pair_reliability = np.sqrt(
            spot_reliability[method_idx, :, None] * spot_reliability[method_idx, None, :]
        ).astype(np.float32)
        pair_mean += pair_reliability / n_methods
        pair_second_moment += (pair_reliability * pair_reliability) / n_methods

    reliability_dispersion = np.sqrt(
        np.maximum(pair_second_moment - pair_mean * pair_mean, 0.0)
    )
    low, high = np.quantile(reliability_dispersion, [0.10, 0.90])
    dispersion_signal = np.clip(
        (reliability_dispersion - low) / (high - low + 1e-12),
        0.0,
        1.0,
    )

    base_conflict = 4.0 * plain_affinity * (1.0 - plain_affinity)
    gate = 1.0 - (1.0 - dispersion_signal) * (1.0 - base_conflict)
    np.fill_diagonal(gate, 1.0)
    return gate


def gated_affinity(
    plain_affinity: np.ndarray,
    weighted_affinity: np.ndarray,
    gate: np.ndarray,
) -> np.ndarray:
    """Fuse plain co-association and reliability-weighted affinity."""
    affinity = plain_affinity + gate * (weighted_affinity - plain_affinity)
    return normalize_affinity(affinity)


def consensus_assignment(affinity: np.ndarray, n_clusters: int) -> np.ndarray:
    """Generate consensus labels with average-linkage agglomerative clustering."""
    distance = 1.0 - np.clip(normalize_affinity(affinity), 0.0, 1.0)
    try:
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage="average",
        )
    return clusterer.fit_predict(distance)


def spatial_neighbors(spatial_coordinates: np.ndarray, k: int) -> np.ndarray:
    """Return k-nearest spatial neighbors for every spot."""
    coordinates = np.asarray(spatial_coordinates, dtype=float)
    k = min(int(k), max(len(coordinates) - 1, 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nbrs.fit(coordinates)
    return nbrs.kneighbors(coordinates, return_distance=False)[:, 1:]


def affinity_confidence(affinity: np.ndarray) -> np.ndarray:
    """Convert pairwise affinity uncertainty into a spot-level confidence score."""
    matrix = normalize_affinity(affinity)
    confidence = 1.0 - (matrix * (1.0 - matrix)).mean(axis=1)
    return (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-12)


def neighborhood_label_entropy(labels: Sequence[int], neighbors: np.ndarray) -> np.ndarray:
    """Measure local label inconsistency in the spatial neighbor graph."""
    values = np.asarray(labels)
    entropy = np.zeros(len(values), dtype=float)

    for idx in range(len(values)):
        _, counts = np.unique(values[neighbors[idx]], return_counts=True)
        probabilities = counts / counts.sum()
        entropy[idx] = -np.sum(probabilities * np.log(probabilities + 1e-12))

    return entropy


def conservative_spatial_refinement(
    labels: Sequence[int],
    affinity: np.ndarray,
    spatial_coordinates: np.ndarray,
    config: TRUSTSTConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Refine only low-confidence spots with strong neighborhood support."""
    labels = np.asarray(labels)
    neighbors = spatial_neighbors(spatial_coordinates, config.spatial_refine_k)
    confidence = affinity_confidence(affinity)
    entropy = neighborhood_label_entropy(labels, neighbors)

    candidates = (
        confidence < np.quantile(confidence, config.confidence_quantile)
    ) | (
        entropy > np.quantile(entropy, config.local_entropy_quantile)
    )

    refined = labels.copy()
    denominator = float(neighbors.shape[1])

    for idx in np.where(candidates)[0]:
        values, counts = np.unique(labels[neighbors[idx]], return_counts=True)
        best = int(np.argmax(counts))
        if counts[best] / denominator >= config.majority_support:
            refined[idx] = values[best]

    diagnostics = {
        "candidate_fraction": float(np.mean(candidates)),
        "changed_fraction": float(np.mean(refined != labels)),
        "mean_affinity_confidence": float(np.mean(confidence)),
    }
    return refined, diagnostics


class TRUSTSTConsensus:
    """Multi-level reliability-guided consensus clustering for spatial transcriptomics."""

    def __init__(
        self,
        config: Optional[TRUSTSTConfig] = None,
        use_method_decorrelation: bool = True,
        use_cluster_reliability: bool = True,
        use_adaptive_consensus_gate: bool = True,
        use_spatial_refinement: bool = True,
    ) -> None:
        self.config = config or TRUSTSTConfig()
        self.use_method_decorrelation = use_method_decorrelation
        self.use_cluster_reliability = use_cluster_reliability
        self.use_adaptive_consensus_gate = use_adaptive_consensus_gate
        self.use_spatial_refinement = use_spatial_refinement

        self.labels_: Optional[np.ndarray] = None
        self.affinity_: Optional[np.ndarray] = None
        self.diagnostics_: Dict[str, object] = {}

    def fit_predict(
        self,
        base_partitions: LabelCollection,
        spatial_coordinates: np.ndarray,
        n_clusters: int,
    ) -> np.ndarray:
        labels, _, _ = self.fit(base_partitions, spatial_coordinates, n_clusters)
        return labels

    def fit(
        self,
        base_partitions: LabelCollection,
        spatial_coordinates: np.ndarray,
        n_clusters: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        partitions, coordinates = validate_inputs(base_partitions, spatial_coordinates)
        n_methods = len(partitions)

        plain_affinity = plain_coassociation_matrix(partitions)

        if self.use_method_decorrelation:
            method_weights, method_similarity, raw_decorrelation_scores = method_decorrelation(
                partitions,
                ridge=self.config.decorrelation_ridge,
                alpha=self.config.decorrelation_alpha,
            )
        else:
            method_weights = np.full(n_methods, 1.0 / n_methods, dtype=float)
            method_similarity = np.eye(n_methods, dtype=float)
            raw_decorrelation_scores = np.ones(n_methods, dtype=float)

        if self.use_cluster_reliability:
            spot_reliability, unit_reliability = cluster_reliability(
                partitions,
                theta=self.config.cluster_reliability_theta,
            )
        else:
            spot_reliability = np.ones((n_methods, len(partitions[0])), dtype=float)
            unit_reliability = {}

        weighted_affinity = reliability_weighted_affinity(
            partitions,
            method_weights,
            spot_reliability,
        )

        if self.use_adaptive_consensus_gate:
            gate = adaptive_consensus_gate(spot_reliability, plain_affinity)
            final_affinity = gated_affinity(plain_affinity, weighted_affinity, gate)
            mean_gate = float(gate.mean())
        else:
            gate = np.ones_like(plain_affinity)
            final_affinity = weighted_affinity
            mean_gate = None

        consensus_labels = consensus_assignment(final_affinity, n_clusters)

        if self.use_spatial_refinement:
            final_labels, refinement = conservative_spatial_refinement(
                consensus_labels,
                final_affinity,
                coordinates,
                self.config,
            )
        else:
            final_labels = consensus_labels
            refinement = {
                "candidate_fraction": 0.0,
                "changed_fraction": 0.0,
                "mean_affinity_confidence": float(affinity_confidence(final_affinity).mean()),
            }

        mean_similarity = float(
            (method_similarity.sum() - n_methods) / max(n_methods * (n_methods - 1), 1)
        )
        diagnostics: Dict[str, object] = {
            "config": self.config.to_dict(),
            "method_weights": method_weights.tolist(),
            "method_similarity": method_similarity.tolist(),
            "raw_decorrelation_scores": raw_decorrelation_scores.tolist(),
            "mean_method_similarity": mean_similarity,
            "mean_cluster_reliability": float(spot_reliability.mean()),
            "mean_adaptive_gate": mean_gate,
            "n_cluster_units": len(unit_reliability),
            "refinement": refinement,
        }

        self.labels_ = final_labels
        self.affinity_ = final_affinity
        self.diagnostics_ = diagnostics
        return final_labels, final_affinity, diagnostics

    @classmethod
    def ablation_variants(
        cls,
        config: Optional[TRUSTSTConfig] = None,
    ) -> Mapping[str, "TRUSTSTConsensus"]:
        """Return the full model and the standard component ablations."""
        return {
            "Full TRUST-ST": cls(config=config),
            "w/o Method Decorrelation": cls(
                config=config,
                use_method_decorrelation=False,
            ),
            "w/o Cluster Reliability": cls(
                config=config,
                use_cluster_reliability=False,
            ),
            "w/o Adaptive Consensus Gate": cls(
                config=config,
                use_adaptive_consensus_gate=False,
            ),
            "w/o Spatial Refinement": cls(
                config=config,
                use_spatial_refinement=False,
            ),
        }
