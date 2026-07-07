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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
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


class FBCSP(BaseEstimator, TransformerMixin):
    """Filter Bank Common Spatial Pattern feature extractor.

    Fits an independent multiclass CSP in each sub-band and concatenates the
    log-variance features. Feature selection (MIBIF-style) is left to a following
    ``SelectKBest`` step in the pipeline.
    """

    def __init__(self, sfreq: float, bands=DEFAULT_BANDS, n_components: int = 4):
        self.sfreq = sfreq
        self.bands = bands
        self.n_components = n_components

    def fit(self, X, y):  # noqa: D102
        self.csps_ = []
        for low, high in self.bands:
            Xb = _bandpass(X, self.sfreq, low, high)
            csp = CSP(n_components=self.n_components, reg="ledoit_wolf", log=True, norm_trace=False)
            csp.fit(Xb, y)
            self.csps_.append(csp)
        return self

    def transform(self, X):  # noqa: D102
        feats = []
        for (low, high), csp in zip(self.bands, self.csps_):
            Xb = _bandpass(X, self.sfreq, low, high)
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
    "fbcsp": lambda sfreq, rs: Pipeline(
        [
            ("fbcsp", FBCSP(sfreq, n_components=4)),
            ("select", SelectKBest(mutual_info_classif, k=24)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    ),
    "riemann": lambda sfreq, rs: _riemann_lda(sfreq, RIEMANN_BANDS, rs),
    "riemann_wide": lambda sfreq, rs: _riemann_lda(sfreq, DEFAULT_BANDS, rs),
    "riemann_lr": lambda sfreq, rs: _riemann_lr(sfreq, RIEMANN_BANDS, rs),
    "riemann_wide_lr": lambda sfreq, rs: _riemann_lr(sfreq, DEFAULT_BANDS, rs),
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
}


def build_decoder(name: str, sfreq: float, random_state: int = 42) -> Pipeline:
    """Return a fresh, unfitted decoding pipeline for the given name."""
    try:
        factory = _DECODER_FACTORIES[name]
    except KeyError:
        raise ValueError(f"Unknown decoder: {name}") from None
    return factory(sfreq, random_state)


DECODER_NAMES = tuple(_DECODER_FACTORIES)
