import pickle
from importlib.metadata import EntryPoint, EntryPoints

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from model_interpreter import interpreter
from model_interpreter.interpreter import ModelInterpreter


class RecordingExplainer:
    """Explainer that records its calls and returns a canned answer."""

    def __init__(self):
        self.calls = []

    def explain(self, claim_features, shap_values, predicted_fraud):
        self.calls.append(
            {
                "claim_features": claim_features,
                "shap_values": shap_values,
                "predicted_fraud": predicted_fraud,
            }
        )
        return f"explanation {len(self.calls)}"


# Explainer returned by the fake entry point below.
REGISTERED_EXPLAINER = RecordingExplainer()


def get_registered_explainer():
    """Factory the fake entry point points at."""
    return REGISTERED_EXPLAINER


def fake_entry_points(*registered):
    """Return an entry_points replacement that only knows ``registered``."""

    def _entry_points(group):
        return EntryPoints(ep for ep in registered if ep.group == group)

    return _entry_points


DEFAULT_ENTRY_POINT = EntryPoint(
    name=interpreter.LLM_EXPLAINER_ENTRY_POINT_NAME,
    value=f"{__name__}:get_registered_explainer",
    group=interpreter.LLM_EXPLAINER_ENTRY_POINT_GROUP,
)


@pytest.fixture
def fitted():
    """ModelInterpreter fitted to a small tree model, with one row."""
    X = pd.DataFrame(
        {
            "Age": [20, 30, 40, 50, 60, 70, 25, 35],
            "Income": [10, 20, 30, 40, 50, 60, 15, 25],
        }
    )
    y = [0, 0, 1, 1, 1, 1, 0, 0]
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    mi = ModelInterpreter(feature_names=list(X.columns))
    mi.fit(model)
    return mi, X.iloc[[2]]


@pytest.fixture(autouse=True)
def reset_registered_explainer():
    REGISTERED_EXPLAINER.calls.clear()


class TestExplainWithLLM(object):
    """Tests for ModelInterpreter.explain_with_llm()."""

    def test_passes_inputs_to_explainer(self, fitted) -> None:
        """Test the transform output, claim and prediction reach the explainer"""
        mi, row = fitted
        contributions = mi.transform(row, return_type="name_value_dicts")
        claim = {"Age": 40, "Income": 30}
        explainer = RecordingExplainer()

        result = mi.explain_with_llm(contributions, claim, 1, explainer)

        assert result == "explanation 1"
        assert explainer.calls == [
            {
                "claim_features": claim,
                "shap_values": contributions,
                "predicted_fraud": 1,
            }
        ]

    def test_returns_explainer_result(self, fitted) -> None:
        """Test the explainer's return value is returned unchanged, e.g. None"""

        class SkipExplainer:
            def explain(self, claim_features, shap_values, predicted_fraud):
                return None

        mi, row = fitted
        assert mi.explain_with_llm(mi.transform(row), {}, 0, SkipExplainer()) is None

    def test_explainer_errors_propagate(self, fitted) -> None:
        """Test errors raised by the explainer reach the caller"""

        class FailingExplainer:
            def explain(self, claim_features, shap_values, predicted_fraud):
                raise RuntimeError("LLM unavailable")

        mi, row = fitted
        with pytest.raises(RuntimeError, match="LLM unavailable"):
            mi.explain_with_llm(mi.transform(row), {}, 1, FailingExplainer())

    def test_given_explainer_skips_entry_point(self, fitted, monkeypatch) -> None:
        """Test no entry point lookup happens when an explainer is passed"""

        def no_lookup(group):
            raise AssertionError("entry points should not be looked up")

        monkeypatch.setattr(interpreter, "entry_points", no_lookup)
        mi, row = fitted
        assert mi.explain_with_llm(mi.transform(row), {}, 1, RecordingExplainer())

    def test_default_from_entry_point(self, fitted, monkeypatch) -> None:
        """Test the registered default explainer is used when none is passed"""
        monkeypatch.setattr(
            interpreter, "entry_points", fake_entry_points(DEFAULT_ENTRY_POINT)
        )
        mi, row = fitted
        contributions = mi.transform(row)

        result = mi.explain_with_llm(contributions, {"Age": 40}, 1)

        assert result == "explanation 1"
        assert REGISTERED_EXPLAINER.calls[0]["shap_values"] == contributions

    def test_no_default_registered(self, fitted, monkeypatch) -> None:
        """Test a clear ImportError when no explainer is passed or registered"""
        other = EntryPoint(
            name="other",
            value=f"{__name__}:get_registered_explainer",
            group=interpreter.LLM_EXPLAINER_ENTRY_POINT_GROUP,
        )
        monkeypatch.setattr(interpreter, "entry_points", fake_entry_points(other))
        mi, row = fitted

        with pytest.raises(ImportError, match="No LLM explainer is installed"):
            mi.explain_with_llm(mi.transform(row), {}, 1)

    def test_no_llm_dependency(self) -> None:
        """Test the interpreter module loads no LLM library"""
        assert not hasattr(interpreter, "ShapLLMExplainer")
        source = open(interpreter.__file__).read()
        assert "openai" not in source

    def test_interpreter_stays_picklable(self, fitted) -> None:
        """Test the explainer is not stored on the interpreter"""
        mi, row = fitted
        explainer = RecordingExplainer()
        mi.explain_with_llm(mi.transform(row), {}, 1, explainer)

        assert explainer not in vars(mi).values()
        # round trip of an object created in this test, not untrusted data
        pickle.loads(pickle.dumps(mi))
