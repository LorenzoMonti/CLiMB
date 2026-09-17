import pathlib
import re
import unittest
import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from CLiMB.core.CLiMB import CLiMB
from CLiMB.core.KBound import KBound
from CLiMB.exploratory.DBSCANExploratory import DBSCANExploratory
from CLiMB.explain import (
    cluster_signatures,
    plot_cluster_signatures,
    plot_feature_attribution,
    plot_gate_accounting,
    plot_heatmap,
    plot_margin_map,
    plot_roles,
)
from CLiMB.utils.util import cohens_d


def _fitted_climb(n_samples=400):
    X, y = make_blobs(n_samples=n_samples, centers=5, n_features=3, random_state=7)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    seeds = {
        tuple(X_scaled[y == c].mean(axis=0)):
            [X_scaled[np.where(y == c)[0][i]].tolist() for i in range(4)]
        for c in range(3)
    }
    climb = CLiMB(
        constrained_clusters=3, seed_points=seeds, density_threshold=0.03,
        distance_threshold=1.2, radial_threshold=2.0, convergence_tolerance=1e-6,
        exploratory_algorithm=DBSCANExploratory(eps=0.35, min_samples=5),
    ).fit(X_scaled)
    return climb, X, X_scaled, scaler


class TestCLiMBExplain(unittest.TestCase):
    """The two phases joined into one table, both reconstructions verified."""

    def setUp(self):
        np.random.seed(42)
        self.climb, self.X, self.X_scaled, self.scaler = _fitted_climb()
        self.names = ["a", "b", "c"]

    def test_explain_covers_every_point_and_splits_by_phase(self):
        table = self.climb.explain(self.X_scaled, feature_names=self.names)

        self.assertEqual(len(table), len(self.X_scaled))
        self.assertEqual(table["point_index"].tolist(), list(range(len(self.X_scaled))))
        np.testing.assert_array_equal(table["final_label"], self.climb.get_labels())

        expected_phase = np.where(self.climb.constrained_labels != -1, 1, 2)
        np.testing.assert_array_equal(table["phase"], expected_phase)
        # Non-degenerate: both phases must have work to show.
        self.assertGreater((table["phase"] == 1).sum(), 0)
        self.assertGreater((table["phase"] == 2).sum(), 0)

    def test_phase2_columns_are_empty_where_phase2_never_looked(self):
        table = self.climb.explain(self.X_scaled, feature_names=self.names)
        kept = table["phase"] == 1

        self.assertTrue(table.loc[kept, "phase2_core_distance"].isna().all())
        self.assertTrue(table.loc[~kept, "phase2_core_distance"].notna().all())
        self.assertTrue(table.loc[kept, "phase2_role"].isna().all())
        self.assertTrue(table.loc[~kept, "phase2_role"].notna().all())

    def test_phase1_columns_reproduce_decision_path(self):
        table = self.climb.explain(self.X_scaled, feature_names=self.names)
        decisions = self.climb.kbound_.decision_path(self.X_scaled)

        for column in decisions.columns:
            if column == "point_index":
                continue
            np.testing.assert_array_equal(
                table[f"phase1_{column}"].to_numpy(), decisions[column].to_numpy(),
                err_msg=f"phase1_{column} diverged from decision_path",
            )
        self.assertEqual(self.climb.kbound_.fidelity_, 1.0)

    def test_scaler_only_changes_the_reported_features(self):
        scaled = self.climb.explain(self.X_scaled, feature_names=self.names)
        physical = self.climb.explain(self.X_scaled, feature_names=self.names,
                                      scaler=self.scaler)

        np.testing.assert_allclose(physical[self.names].to_numpy(), self.X)
        np.testing.assert_allclose(scaled[self.names].to_numpy(), self.X_scaled)
        # The explanation itself is computed in the fitted space either way.
        np.testing.assert_array_equal(scaled["phase1_gate"], physical["phase1_gate"])
        np.testing.assert_allclose(scaled["phase1_distance_margin"],
                                   physical["phase1_distance_margin"])

    def test_explain_rejects_bad_input(self):
        with self.assertRaises(RuntimeError):
            CLiMB(constrained_clusters=3).explain(self.X_scaled)
        with self.assertRaises(ValueError):
            self.climb.explain(self.X_scaled, feature_names=["only", "two"])
        with self.assertRaises(ValueError):
            self.climb.explain(self.X_scaled[:10])

    def test_default_feature_names(self):
        table = self.climb.explain(self.X_scaled)
        for j in range(self.X_scaled.shape[1]):
            self.assertIn(f"feature_{j}", table.columns)


class TestFeatureAttribution(unittest.TestCase):
    """
    An exact algebraic split of the metric, not an attribution model: the shares
    must add back up to the distance they came from.
    """

    def setUp(self):
        np.random.seed(42)
        X, y = make_blobs(n_samples=300, centers=3, n_features=3, random_state=0)
        self.X = StandardScaler().fit_transform(X)
        self.seeds = {
            tuple(self.X[y == c].mean(axis=0) + 0.5):
                [self.X[np.where(y == c)[0][i]].tolist() for i in range(3)]
            for c in range(3)
        }
        self.params = dict(
            n_clusters=3, seeds=self.seeds, density_threshold=0.02,
            distance_threshold=1.5, radial_threshold=50.0,
            convergence_tolerance=1e-9, max_iter=300,
        )
        self.names = ["a", "b", "c"]

    def _metrics(self):
        return {
            "euclidean": dict(distance_metric="euclidean"),
            "mahalanobis": dict(distance_metric="mahalanobis",
                                metric_params={"VI": np.linalg.inv(np.cov(self.X.T))}),
        }

    def test_contributions_sum_back_to_the_distance(self):
        for name, extra in self._metrics().items():
            with self.subTest(metric=name):
                kbound = KBound(**self.params, **extra).fit(self.X)
                table = kbound.feature_attribution(self.X, feature_names=self.names)

                contributions = table[[f"contrib_{n}" for n in self.names]].sum(axis=1)
                np.testing.assert_allclose(contributions, table["d2_reference"])

                discriminative = table[[f"disc_{n}" for n in self.names]].sum(axis=1)
                np.testing.assert_allclose(discriminative, table["margin2"])
                np.testing.assert_allclose(
                    table["margin2"], table["d2_competitor"] - table["d2_reference"]
                )

    def test_attribution_agrees_with_decision_path(self):
        kbound = KBound(**self.params).fit(self.X)
        decisions = kbound.decision_path(self.X)
        table = kbound.feature_attribution(self.X, feature_names=self.names)

        assigned = (decisions["gate"] == "assigned").to_numpy()
        self.assertGreater(assigned.sum(), 0)
        np.testing.assert_allclose(
            table.loc[assigned, "d2_reference"],
            decisions.loc[assigned, "distance_to_nearest"] ** 2,
        )

    def test_shares_are_fractions_of_their_own_total(self):
        kbound = KBound(**self.params).fit(self.X)
        table = kbound.feature_attribution(self.X, feature_names=self.names)
        shares = table[[f"contrib_{n}_frac" for n in self.names]].sum(axis=1)
        np.testing.assert_allclose(shares.dropna(), 1.0)

    def test_custom_metric_has_no_additive_split(self):
        """
        A custom distance is not a quadratic form, so there is no exact
        decomposition. Refusing beats returning numbers that look like one.
        """
        from scipy.spatial.distance import euclidean

        kbound = KBound(distance_metric="custom", metric_params={"func": euclidean},
                        **self.params).fit(self.X)
        with self.assertRaises(RuntimeError):
            kbound.feature_attribution(self.X)
        # The gate-level explanation still works for any metric.
        self.assertEqual(len(kbound.decision_path(self.X)), len(self.X))

    def test_attribution_rejects_bad_input(self):
        kbound = KBound(**self.params).fit(self.X)
        with self.assertRaises(ValueError):
            kbound.feature_attribution(self.X, feature_names=["a", "b"])
        with self.assertRaises(ValueError):
            kbound.feature_attribution(self.X[:10])
        with self.assertRaises(RuntimeError):
            KBound(**self.params).feature_attribution(self.X)


class TestDescriptiveStatistics(unittest.TestCase):
    """
    Kept apart from the reconstructions on purpose: these are computed after the
    fact by our rules, and nothing verifies them against the model.
    """

    def setUp(self):
        np.random.seed(42)
        self.rng = np.random.RandomState(0)
        self.X = np.vstack([self.rng.normal(0, 1, (60, 3)),
                            self.rng.normal(3, 1, (60, 3)),
                            self.rng.normal(0, 1, (40, 3))])
        self.labels = np.array([0] * 60 + [1] * 60 + [-1] * 40)
        self.names = ["a", "b", "c"]

    def test_cohens_d_sign_and_magnitude(self):
        low = self.rng.normal(0, 1, 300)
        high = self.rng.normal(3, 1, 300)
        self.assertLess(cohens_d(low, high), -2)
        self.assertGreater(cohens_d(high, low), 2)
        self.assertLess(abs(cohens_d(low, self.rng.normal(0, 1, 300))), 0.3)

    def test_cohens_d_is_nan_when_it_cannot_be_computed(self):
        self.assertTrue(np.isnan(cohens_d([1.0], [1.0, 2.0, 3.0])))
        self.assertTrue(np.isnan(cohens_d([2.0, 2.0, 2.0], [2.0, 2.0, 2.0])))
        self.assertTrue(np.isnan(cohens_d([np.nan, np.nan], [1.0, 2.0, 3.0])))
        # NaNs are dropped, not propagated, when enough values remain.
        self.assertFalse(np.isnan(cohens_d([1.0, 2.0, np.nan, 3.0], [5.0, 6.0, 7.0])))

    def test_cluster_signatures_shape_and_contrast(self):
        signatures = cluster_signatures(self.X, self.labels, feature_names=self.names)

        self.assertEqual(list(signatures.columns), self.names)
        self.assertEqual(sorted(signatures.index), [0, 1])
        self.assertNotIn(-1, signatures.index)
        # Cluster 1 is shifted away from the noise reference, cluster 0 is not.
        self.assertGreater(signatures.loc[1].abs().min(), 1.0)
        self.assertLess(signatures.loc[0].abs().max(), 1.0)

    def test_cluster_signatures_warns_when_the_reference_is_too_small(self):
        labels = np.where(np.arange(len(self.X)) == 0, -1, 0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            signatures = cluster_signatures(self.X, labels, feature_names=self.names)
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))
        self.assertTrue(signatures.isna().all().all())

    def test_cluster_signatures_accepts_an_explicit_reference(self):
        local = cluster_signatures(self.X, self.labels, feature_names=self.names)
        absolute = cluster_signatures(self.X, self.labels, reference=self.X,
                                      feature_names=self.names)
        self.assertEqual(local.shape, absolute.shape)
        self.assertFalse(np.allclose(local.to_numpy(), absolute.to_numpy()))

    def test_cluster_signatures_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            cluster_signatures(self.X, self.labels[:10])
        with self.assertRaises(ValueError):
            cluster_signatures(self.X, self.labels, feature_names=["a"])
        with self.assertRaises(ValueError):
            cluster_signatures(self.X, self.labels, reference=self.X[:, :2])


class TestPlotHelpers(unittest.TestCase):
    """Domain-neutral: every label the reader sees comes from the caller."""

    def setUp(self):
        np.random.seed(42)
        self.climb, self.X, self.X_scaled, self.scaler = _fitted_climb()
        self.names = ["a", "b", "c"]
        self.table = self.climb.explain(self.X_scaled, feature_names=self.names)

    def test_axis_and_colourbar_text_is_the_callers(self):
        axes = plot_margin_map(
            self.table[self.table["phase"] == 1], x="a", y="b",
            value="phase1_distance_margin",
            xlabel="Lz (10³ kpc km/s)", ylabel="Energy (10⁵ km²/s²)",
            colourbar_label="distance to decision boundary",
        )
        self.assertEqual(axes.get_xlabel(), "Lz (10³ kpc km/s)")
        self.assertEqual(axes.get_ylabel(), "Energy (10⁵ km²/s²)")

    def test_labels_default_to_the_column_names(self):
        axes = plot_margin_map(self.table, x="a", y="b", value="phase1_local_density")
        self.assertEqual(axes.get_xlabel(), "a")
        self.assertEqual(axes.get_ylabel(), "b")

    def test_gate_accounting_counts_every_point(self):
        axes = plot_gate_accounting(self.table, column="phase1_gate")
        total = sum(bar.get_height() for bar in axes.patches)
        self.assertEqual(int(total), len(self.table))

    def test_roles_plot_renders_the_phase2_table(self):
        """
        Plot against phase2_label, not phase2_algorithm_cluster: a name map is
        keyed on CLiMB's numbering, and the algorithm's own starts back at 0
        where Phase 1's names live.
        """
        phase2 = self.table[self.table["phase"] == 2]
        discovery = int(phase2.loc[phase2["phase2_label"] != -1, "phase2_label"].iloc[0])
        axes = plot_roles(phase2, x="a", y="b", role_column="phase2_role",
                          cluster_column="phase2_label",
                          cluster_names={discovery: "a name the package cannot know"})
        self.assertTrue(axes.get_legend() is not None)

        legend = [text.get_text() for text in axes.get_legend().get_texts()]
        self.assertTrue(any("a name the package cannot know" in entry for entry in legend))

    def test_heatmap_survives_an_all_nan_matrix(self):
        """An effect size with too little data is NaN, and that must still draw."""
        axes = plot_heatmap(np.full((2, 3), np.nan), ["a", "b", "c"], ["x", "y"],
                            colourbar_label="effect size")
        self.assertEqual(axes.images[0].get_clim(), (-1.0, 1.0))

    def test_attribution_and_signature_heatmaps_render(self):
        attribution = self.climb.kbound_.feature_attribution(
            self.X_scaled, feature_names=self.names
        )
        self.assertIsNotNone(plot_feature_attribution(attribution, self.names))
        self.assertIsNotNone(plot_feature_attribution(attribution, self.names,
                                                      kind="contrib"))
        with self.assertRaises(ValueError):
            plot_feature_attribution(attribution, self.names, kind="nonsense")

        signatures = cluster_signatures(
            self.climb.unassigned_points,
            np.asarray(self.climb.exploratory_labels),
            reference=self.X_scaled, feature_names=self.names,
        )
        self.assertIsNotNone(plot_cluster_signatures(signatures))

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")


class TestPackageStaysDomainNeutral(unittest.TestCase):
    """
    The interpretability layer came from an astrophysics project. Units, object
    names and field conventions belong to the caller; if one leaks back into the
    package it will quietly mislabel someone else's axes.
    """

    # Whole words only: "start" contains "star", and a substring match would
    # make this test cry wolf until someone switched it off.
    FORBIDDEN_WORDS = ("kpc", "lperp", "gaia", "star", "stars", "stellar",
                       "galactic", "galaxy", "astrophysical", "substructure")
    FORBIDDEN_SYMBOLS = ("km/s", "km²/s²", "fe/h")

    def test_no_domain_vocabulary_in_the_package(self):
        package = pathlib.Path(__file__).resolve().parent.parent / "CLiMB"
        pattern = re.compile(r"\b(" + "|".join(self.FORBIDDEN_WORDS) + r")\b")

        offences = []
        for source in sorted(package.rglob("*.py")):
            for line_number, line in enumerate(
                source.read_text(encoding="utf-8").splitlines(), start=1
            ):
                lowered = line.lower()
                hits = set(pattern.findall(lowered))
                hits |= {s for s in self.FORBIDDEN_SYMBOLS if s in lowered}
                for hit in sorted(hits):
                    offences.append(
                        f"{source.name}:{line_number}: {hit!r} in {line.strip()!r}"
                    )

        self.assertEqual(offences, [], "domain vocabulary leaked into the package:\n"
                                       + "\n".join(offences))

    def test_the_neutrality_check_can_actually_fail(self):
        """A guard that cannot fire is not a guard."""
        pattern = re.compile(r"\b(" + "|".join(self.FORBIDDEN_WORDS) + r")\b")
        self.assertTrue(pattern.search('ax.set_xlabel("Lz (10³ kpc km/s)")'.lower()))
        self.assertTrue(pattern.search("# one row per star".lower()))
        self.assertFalse(pattern.search("labels might start from 3".lower()))
        self.assertFalse(pattern.search("the walk restarted here".lower()))


class TestLabelNumberingIsUnambiguous(unittest.TestCase):
    """
    Phase 2's algorithm numbers its clusters from 0 and CLiMB offsets them, so
    the same table carries two numbering systems. They overlap, which means
    confusing them does not raise -- it produces a plausible, wrong answer. A
    name map built for Phase 1 will happily match the algorithm's cluster 0 and
    label a discovery with a constrained cluster's name.

    The column names are the only thing standing between a reader and that
    mistake, so they are pinned here.
    """

    def setUp(self):
        np.random.seed(42)
        X, y = make_blobs(n_samples=600, centers=6, n_features=3, random_state=7)
        scaler = StandardScaler()
        self.X = scaler.fit_transform(X)
        seeds = {
            tuple(self.X[y == c].mean(axis=0)):
                [self.X[np.where(y == c)[0][i]].tolist() for i in range(4)]
            for c in range(3)
        }
        self.climb = CLiMB(
            constrained_clusters=3, seed_points=seeds, density_threshold=0.03,
            distance_threshold=1.0, radial_threshold=2.0, convergence_tolerance=1e-6,
            exploratory_algorithm=DBSCANExploratory(eps=0.30, min_samples=5),
        ).fit(self.X)
        self.table = self.climb.explain(self.X)
        self.phase2 = self.table[self.table["phase"] == 2]

    def test_the_two_numberings_really_do_collide(self):
        """Without this, the rest of the class could pass for the wrong reason."""
        algorithm = set(self.phase2["phase2_algorithm_cluster"].dropna())
        phase1 = set(self.table.loc[self.table["phase"] == 1, "phase1_label"])
        self.assertTrue(algorithm & phase1,
                        "the numberings no longer overlap, so this guard is moot")

    def test_phase_label_columns_agree_with_final_label(self):
        """phase1_label and phase2_label mean the same thing, so they read alike."""
        phase1 = self.table[self.table["phase"] == 1]
        np.testing.assert_array_equal(phase1["phase1_label"], phase1["final_label"])
        np.testing.assert_array_equal(
            self.phase2["phase2_label"].to_numpy(),
            self.phase2["final_label"].to_numpy().astype(float),
        )

    def test_the_algorithms_own_numbering_is_named_for_it(self):
        self.assertIn("phase2_algorithm_cluster", self.table.columns)
        # The old name said "cluster" without saying whose, which is what let it
        # be read as a CLiMB label.
        self.assertNotIn("phase2_cluster", self.table.columns)

        raw = self.climb.exploratory_algorithm.explain(self.climb.unassigned_points)
        np.testing.assert_array_equal(
            self.phase2["phase2_algorithm_cluster"].to_numpy(),
            raw["cluster"].to_numpy().astype(float),
        )

    def test_the_offset_is_what_separates_the_two(self):
        clustered = self.phase2["phase2_algorithm_cluster"] != -1
        offset = (self.phase2.loc[clustered, "phase2_label"]
                  - self.phase2.loc[clustered, "phase2_algorithm_cluster"])
        self.assertEqual(offset.nunique(), 1, "the offset must be one constant")
        self.assertGreater(offset.iloc[0], 0)
        # Noise stays -1 in both, rather than being shifted.
        noise = ~clustered
        self.assertTrue((self.phase2.loc[noise, "phase2_label"] == -1).all())
