"""Decoding pipelines for BCI Competition IV 2a.

This module provides the decoders used for the competition-style benchmark:

- ``csp_lda``   : single-band (8-30 Hz) CSP + LDA baseline.
- ``fbcsp``     : Filter Bank CSP + mutual-information feature selection + LDA.
                  This reproduces the family of methods that won the 2008
                  competition (Ang et al., FBCSP, kappa ~0.57).
- ``riemann``   : spatial-covariance + Riemannian tangent space + logistic
                  regression, a strong modern classical decoder.

All decoders expect broadband epochs shaped ``(trials, channels, samples)`` and
apply their own band-pass filtering internally, so a single broadband epoch set
(e.g. 4-40 Hz) can feed every decoder.
"""
from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Motor-relevant sub-bands (mu + beta) used by the filter-bank tangent-space
# decoder. Splitting 8-30 Hz into narrow bands lets the tangent-space features
# capture band-specific spatial covariance structure, which is the standard way
# to push Riemannian decoding above the single-band baseline on 2a.
RIEMANN_BANDS: tuple[tuple[float, float], ...] = (
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 30),
)

# Standard FBCSP filter bank: nine overlapping 4 Hz-wide sub-bands from 4-40 Hz.
DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 28),
    (28, 32),
    (32, 36),
    (36, 40),
)


def _bandpass(X: np.ndarray, sfreq: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last (time) axis."""
    nyq = sfreq / 2.0
    high = min(high, nyq - 1e-3)
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float64)


class BandPass(BaseEstimator, TransformerMixin):
    """Fixed band-pass filter as a scikit-learn transformer step."""

    def __init__(self, sfreq: float, low: float = 8.0, high: float = 30.0):
        self.sfreq = sfreq
        self.low = low
        self.high = high

    def fit(self, X, y=None):  # noqa: D102
        return self

    def transform(self, X):  # noqa: D102
        return _bandpass(np.asarray(X, dtype=np.float64), self.sfreq, self.low, self.high)


def _euclidean_align_ref(X: np.ndarray) -> np.ndarray:
    """Inverse-square-root whitening matrix for Euclidean Alignment (He & Wu 2020).

    ``X`` is ``(trials, channels, samples)``. Transforming each trial as
    ``ref_inv_sqrt @ X_i`` recenters the batch's mean spatial covariance to the
    identity, cancelling session-specific covariance shift.
    """
    from pyriemann.estimation import Covariances
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_covariance

    covs = Covariances(estimator="oas").fit_transform(X)
    return invsqrtm(mean_covariance(covs, metric="euclid"))


def _apply_align(X: np.ndarray, ref_inv_sqrt: np.ndarray) -> np.ndarray:
    return np.einsum("ij,tjk->tik", ref_inv_sqrt, X)


class FBCSP(BaseEstimator, TransformerMixin):
    """Filter Bank Common Spatial Pattern feature extractor.

    Fits an independent multiclass CSP in each sub-band and concatenates the
    log-variance features. Feature selection (MIBIF-style) is left to a following
    ``SelectKBest`` step in the pipeline.
    """

    def __init__(self, sfreq: float, bands=DEFAULT_BANDS, n_components: int = 4, align: str | None = None):
        self.sfreq = sfreq
        self.bands = bands
        self.n_components = n_components
        self.align = align  # None | "euclid" -- signal-space EA before CSP, mirroring riemann_ea

    def fit(self, X, y):  # noqa: D102
        self.csps_ = []
        for low, high in self.bands:
            Xb = _bandpass(X, self.sfreq, low, high)
            if self.align:
                Xb = _apply_align(Xb, _euclidean_align_ref(Xb))
            csp = CSP(n_components=self.n_components, reg="ledoit_wolf", log=True, norm_trace=False)
            csp.fit(Xb, y)
            self.csps_.append(csp)
        return self

    def transform(self, X):  # noqa: D102
        # Transductive, like FilterBankTangentSpace._recenter: the reference is
        # re-estimated from whatever batch is passed, so train (T) and eval (E)
        # each get recentered by their own mean independently. Not valid for
        # single-trial streaming prediction -- see FilterBankTangentSpace docstring.
        feats = []
        for (low, high), csp in zip(self.bands, self.csps_):
            Xb = _bandpass(X, self.sfreq, low, high)
            if self.align:
                Xb = _apply_align(Xb, _euclidean_align_ref(Xb))
            feats.append(csp.transform(Xb))
        return np.concatenate(feats, axis=1)


class FilterBankTangentSpace(BaseEstimator, TransformerMixin):
    """Filter-bank Riemannian tangent-space feature extractor.

    For each sub-band it band-passes the signal, estimates a per-trial spatial
    covariance matrix, and projects it to the Riemannian tangent space. The
    per-band tangent vectors are concatenated into a single feature vector, so a
    plain (shrinkage) linear classifier can exploit band-specific covariance
    structure. This generalises the single-band ``Covariances -> TangentSpace``
    decoder and is the usual way to lift Riemannian decoding on 2a.
    """

    def __init__(self, sfreq: float, bands=RIEMANN_BANDS, estimator: str = "oas", align: str | None = None):
        self.sfreq = sfreq
        self.bands = bands
        self.estimator = estimator
        self.align = align  # None | "euclid" (Euclidean Alignment) | "riemann" (Riemannian Alignment)

    def _recenter(self, covs: np.ndarray) -> np.ndarray:
        """Whiten a batch of covariances by their mean, recentering it to identity.

        This is transductive session alignment (He & Wu 2020): the reference mean
        is computed from *this* batch, so the ``T`` and ``E`` sessions are each
        recentered independently, cancelling the session-specific covariance shift
        that otherwise hurts train-on-T / test-on-E transfer.

        NOTE: because the reference is estimated from the batch, aligned decoders
        (``align`` set) require a representative batch at predict time -- the
        benchmark scores the whole ``E`` set at once. They are NOT valid for
        single-window streaming prediction (demo/live-demo/replay-live), where a
        one-trial batch would collapse every covariance to the identity. Use the
        reference-free ``riemann`` there instead.
        """
        from pyriemann.utils.base import invsqrtm
        from pyriemann.utils.mean import mean_covariance

        ref_inv_sqrt = invsqrtm(mean_covariance(covs, metric=self.align))
        return ref_inv_sqrt @ covs @ ref_inv_sqrt

    def fit(self, X, y=None):  # noqa: D102
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace

        self.pipes_ = []
        for low, high in self.bands:
            Xb = _bandpass(X, self.sfreq, low, high)
            cov = Covariances(estimator=self.estimator)
            covs = cov.fit_transform(Xb)
            if self.align:
                covs = self._recenter(covs)
            ts = TangentSpace().fit(covs)
            self.pipes_.append((cov, ts))
        return self

    def transform(self, X):  # noqa: D102
        feats = []
        for (low, high), (cov, ts) in zip(self.bands, self.pipes_):
            Xb = _bandpass(X, self.sfreq, low, high)
            covs = cov.transform(Xb)
            if self.align:
                covs = self._recenter(covs)
            feats.append(ts.transform(covs))
        return np.concatenate(feats, axis=1)


class FilterBankRiemannClassifier(BaseEstimator, ClassifierMixin):
    """Filter-bank wrapper around pyRiemann covariance classifiers.

    MDM and FgMDM are EEG-specific Riemannian baselines. They classify each
    trial by distances between covariance matrices rather than by tangent-space
    features. We fit one classifier per motor band and average class
    probabilities across bands, matching the filter-bank idea used by FBCSP and
    the tangent-space decoder.
    """

    def __init__(
        self,
        sfreq: float,
        classifier: str = "mdm",
        bands=RIEMANN_BANDS,
        estimator: str = "oas",
        metric: str = "riemann",
    ):
        self.sfreq = sfreq
        self.classifier = classifier
        self.bands = bands
        self.estimator = estimator
        self.metric = metric

    def _make_classifier(self):
        from pyriemann.classification import FgMDM, MDM

        if self.classifier == "mdm":
            return MDM(metric=self.metric)
        if self.classifier == "fgmdm":
            return FgMDM(metric=self.metric)
        raise ValueError(f"Unknown Riemannian classifier: {self.classifier}")

    def fit(self, X, y):  # noqa: D102
        from pyriemann.estimation import Covariances

        self.classes_ = np.unique(y)
        self.pipes_ = []
        for low, high in self.bands:
            Xb = _bandpass(X, self.sfreq, low, high)
            cov = Covariances(estimator=self.estimator)
            covs = cov.fit_transform(Xb)
            clf = self._make_classifier()
            clf.fit(covs, y)
            self.pipes_.append((cov, clf))
        return self

    def predict_proba(self, X):  # noqa: D102
        probs = []
        for (low, high), (cov, clf) in zip(self.bands, self.pipes_):
            Xb = _bandpass(X, self.sfreq, low, high)
            covs = cov.transform(Xb)
            probs.append(clf.predict_proba(covs))
        return np.mean(np.stack(probs), axis=0)

    def predict(self, X):  # noqa: D102
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class KMeansVoteClassifier(BaseEstimator, ClassifierMixin):
    """Classic unsupervised K-Means, evaluated as a classifier via cluster-label voting.

    Unlike ``NearestCentroid`` (which builds one centroid per *known* class
    directly from labels), this runs actual K-Means clustering on the feature
    space with no notion of the four motor-imagery classes -- it just finds
    ``n_clusters`` groups by minimizing within-cluster variance. Labels are
    only used afterwards, to assign each discovered cluster the majority true
    class of the training trials that landed in it (the standard
    "cluster-then-label" way to score unsupervised clustering against ground
    truth). At predict time a trial is assigned to its nearest cluster
    center, then labeled with that cluster's majority class.
    """

    def __init__(self, n_clusters: int = 4, random_state: int = 42, n_init: int = 10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X, y):  # noqa: D102
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=self.n_init)
        cluster_ids = self.kmeans_.fit_predict(X)
        fallback = self.classes_[np.argmax(np.bincount(np.searchsorted(self.classes_, y)))]
        self.cluster_to_label_ = {}
        for c in range(self.n_clusters):
            mask = cluster_ids == c
            if not mask.any():
                self.cluster_to_label_[c] = fallback
                continue
            values, counts = np.unique(y[mask], return_counts=True)
            self.cluster_to_label_[c] = values[np.argmax(counts)]
        return self

    def predict(self, X):  # noqa: D102
        cluster_ids = self.kmeans_.predict(X)
        return np.array([self.cluster_to_label_[c] for c in cluster_ids])


def _riemann_lda(sfreq: float, bands, random_state: int, align: str | None = None) -> Pipeline:
    return Pipeline(
        [
            ("fbts", FilterBankTangentSpace(sfreq, bands, estimator="oas", align=align)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def _riemann_lr(sfreq: float, bands, random_state: int, align: str | None = None) -> Pipeline:
    return Pipeline(
        [
            ("fbts", FilterBankTangentSpace(sfreq, bands, estimator="oas", align=align)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)),
        ]
    )


def _riemann_centroid(sfreq: float, bands, align: str | None = None) -> Pipeline:
    """Nearest-centroid classifier on (optionally aligned) tangent-space features.

    Where MDM clusters by Riemannian distance between covariance matrices, this
    instead computes one Euclidean centroid per class in the *tangent-space*
    feature space (after standardizing) and classifies by nearest centroid --
    a cheap clustering-style baseline to compare against the LDA/LR heads.
    """
    return Pipeline(
        [
            ("fbts", FilterBankTangentSpace(sfreq, bands, estimator="oas", align=align)),
            ("scale", StandardScaler()),
            ("centroid", NearestCentroid()),
        ]
    )


def _riemann_kmeans(sfreq: float, bands, random_state: int, align: str | None = None) -> Pipeline:
    """Classic K-Means clustering (n_clusters=4), scored via cluster-to-label voting."""
    return Pipeline(
        [
            ("fbts", FilterBankTangentSpace(sfreq, bands, estimator="oas", align=align)),
            ("scale", StandardScaler()),
            ("kmeans", KMeansVoteClassifier(n_clusters=4, random_state=random_state)),
        ]
    )


def _fbcsp_lda(sfreq: float, k: int, random_state: int, align: str | None = None) -> Pipeline:
    return Pipeline(
        [
            ("fbcsp", FBCSP(sfreq, n_components=4, align=align)),
            ("select", SelectKBest(mutual_info_classif, k=k)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


# Registry of decoder factories: name -> callable(sfreq, random_state) -> estimator.
# Keeping every decoder here means ``DECODER_NAMES`` (and the CLI choices derived
# from it) stay in sync automatically.
_DECODER_FACTORIES: dict[str, "callable"] = {
    "csp_lda": lambda sfreq, rs: Pipeline(
        [
            ("band", BandPass(sfreq, 8.0, 30.0)),
            ("csp", CSP(n_components=8, reg="ledoit_wolf", log=True, norm_trace=False)),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    ),
    "fbcsp": lambda sfreq, rs: _fbcsp_lda(sfreq, 24, rs),
    # Experiment: fewer MIBIF-selected features than the default k=24.
    "fbcsp_k16": lambda sfreq, rs: _fbcsp_lda(sfreq, 16, rs),
    "fbcsp_k12": lambda sfreq, rs: _fbcsp_lda(sfreq, 12, rs),
    # Experiment: Euclidean Alignment on the raw signal before CSP, mirroring
    # riemann_ea -- FBCSP previously got no session-drift correction at all.
    "fbcsp_ea": lambda sfreq, rs: _fbcsp_lda(sfreq, 24, rs, align="euclid"),
    "riemann": lambda sfreq, rs: _riemann_lda(sfreq, RIEMANN_BANDS, rs),
    "riemann_wide": lambda sfreq, rs: _riemann_lda(sfreq, DEFAULT_BANDS, rs),
    "riemann_lr": lambda sfreq, rs: _riemann_lr(sfreq, RIEMANN_BANDS, rs),
    "riemann_wide_lr": lambda sfreq, rs: _riemann_lr(sfreq, DEFAULT_BANDS, rs),
    "mdm": lambda sfreq, rs: FilterBankRiemannClassifier(sfreq, classifier="mdm"),
    "fgmdm": lambda sfreq, rs: FilterBankRiemannClassifier(sfreq, classifier="fgmdm"),
    # Session-aligned Riemann: recenter each session's covariances before the
    # tangent-space projection to cancel T->E drift (He & Wu 2020).
    "riemann_ea": lambda sfreq, rs: _riemann_lda(sfreq, RIEMANN_BANDS, rs, align="euclid"),
    "riemann_ra": lambda sfreq, rs: _riemann_lda(sfreq, RIEMANN_BANDS, rs, align="riemann"),
    "riemann_fbcsp_vote": lambda sfreq, rs: VotingClassifier(
        estimators=[
            ("riemann", build_decoder("riemann", sfreq, rs)),
            ("fbcsp", build_decoder("fbcsp", sfreq, rs)),
        ],
        voting="soft",
    ),
    # Aligned version of the best ensemble: EA-Riemann soft-voted with FBCSP.
    "riemann_ea_fbcsp_vote": lambda sfreq, rs: VotingClassifier(
        estimators=[
            ("riemann_ea", build_decoder("riemann_ea", sfreq, rs)),
            ("fbcsp", build_decoder("fbcsp", sfreq, rs)),
        ],
        voting="soft",
    ),
    # Same ensemble but with Riemannian (geometric-mean) alignment instead of EA.
    "riemann_ra_fbcsp_vote": lambda sfreq, rs: VotingClassifier(
        estimators=[
            ("riemann_ra", build_decoder("riemann_ra", sfreq, rs)),
            ("fbcsp", build_decoder("fbcsp", sfreq, rs)),
        ],
        voting="soft",
    ),
    # Experiment: weight the best ensemble towards Riemann (0.7) over FBCSP (0.3)
    # instead of the default equal (1, 1) soft vote.
    "riemann_ea_fbcsp_vote_w70": lambda sfreq, rs: VotingClassifier(
        estimators=[
            ("riemann_ea", build_decoder("riemann_ea", sfreq, rs)),
            ("fbcsp", build_decoder("fbcsp", sfreq, rs)),
        ],
        voting="soft",
        weights=[0.7, 0.3],
    ),
    # Experiment: clustering-style classifier -- nearest Euclidean centroid per
    # class on EA-aligned tangent-space features, instead of LDA/LR.
    "riemann_ea_centroid": lambda sfreq, rs: _riemann_centroid(sfreq, RIEMANN_BANDS, align="euclid"),
    # Experiment: classic unsupervised K-Means (no labels used to place cluster
    # centers) on EA-aligned tangent-space features, scored via majority-vote
    # cluster-to-label mapping -- distinct from the supervised centroid above.
    "riemann_ea_kmeans": lambda sfreq, rs: _riemann_kmeans(sfreq, RIEMANN_BANDS, rs, align="euclid"),
    # Experiment: align both ensemble members instead of just the Riemann branch.
    "riemann_ea_fbcsp_ea_vote": lambda sfreq, rs: VotingClassifier(
        estimators=[
            ("riemann_ea", build_decoder("riemann_ea", sfreq, rs)),
            ("fbcsp_ea", build_decoder("fbcsp_ea", sfreq, rs)),
        ],
        voting="soft",
    ),
    # Experiment: three-way vote keeping both the unaligned FBCSP (for the
    # decorrelated errors that made the original vote work) and the aligned
    # fbcsp_ea (for its own standalone gain), alongside riemann_ea.
    "riemann_ea_fbcsp_dual_vote": lambda sfreq, rs: VotingClassifier(
        estimators=[
            ("riemann_ea", build_decoder("riemann_ea", sfreq, rs)),
            ("fbcsp", build_decoder("fbcsp", sfreq, rs)),
            ("fbcsp_ea", build_decoder("fbcsp_ea", sfreq, rs)),
        ],
        voting="soft",
    ),
}


def build_decoder(name: str, sfreq: float, random_state: int = 42) -> Pipeline:
    """Return a fresh, unfitted decoding pipeline for the given name."""
    try:
        factory = _DECODER_FACTORIES[name]
    except KeyError:
        raise ValueError(f"Unknown decoder: {name}") from None
    return factory(sfreq, random_state)


DECODER_NAMES = tuple(_DECODER_FACTORIES)
