MODEL INTERPRETER

----

`model_interpreter` returns feature importance values for a single row model prediction with functionality to:
- handle regression and binary / multiclass classification models
- sort features by importance
- map feature names to more interpretable names
- aggregate feature importance's across features
- handle categorical data within the input features

'model_interpreter' uses [SHAP](https://github.com/shap/shap) to calculate the single row feature importance values.

The package tries to fit one of the three SHAP explainers below in the following order;

- TreeExplainer
- LinearExplainer
- KernelExplainer

Here is a simple example of generating single row feature importance's for a classification model;

```python
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification
from model_interpreter.interpreter import ModelInterpreter

# generate a classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False,
)

# fit a model
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)

# fit model interpreter to model
feature_names = ["feature1", "feature2", "feature3", "feature4"]
model_interpreter = ModelInterpreter(feature_names)
model_interpreter.fit(clf)

# return feature contribution importances for a single row
single_row = X.head(1)
contribution_list = model_interpreter.transform(
    single_row, return_type="name_value_dicts"
)

print(contribution_list)
```

Which will return the following output:
```
[{'Name': 'feature2', 'Value': -0.349129583}, {'Name': 'feature1', 'Value': -0.0039231513}, {'Name': 'feature4', 'Value': 0.0031653932}, {'Name': 'feature3', 'Value': 0.0013787609}]
```

If you also want to include original feature values in the output, use `return_feature_values=True`:
```
contribution_list = model_interpreter.transform(
    single_row, return_type="name_value_dicts", return_feature_values=True
)

print(contribution_list)
```

Which will then return output like:
```
[{'Name': 'feature2', 'Value': -0.349129583, 'FeatureValue': 3.5}, {'Name': 'feature1', 'Value': -0.0039231513, 'FeatureValue': 22.0}, {'Name': 'feature4', 'Value': 0.0031653932, 'FeatureValue': 1.0}, {'Name': 'feature3', 'Value': 0.0013787609, 'FeatureValue': 0.5}]
```


## LLM explanations

`ModelInterpreter.explain_with_llm` turns the output of `transform` for one prediction into a plain-language explanation. `model_interpreter` contains no LLM code and has no LLM dependency: you bring your own explainer, which is any object with an `explain(claim_features, shap_values, predicted_fraud)` method.

```python
class MyExplainer:
    def explain(self, claim_features, shap_values, predicted_fraud):
        prompt = f"Explain prediction {predicted_fraud} for {claim_features} given {shap_values}"
        return my_llm_client(prompt)  # any LLM client or service


contribution_list = model_interpreter.transform(single_row, return_type="name_value_dicts")

explanation = model_interpreter.explain_with_llm(
    contribution_list,
    claim_features=raw_claim_row,  # the observation's own (unencoded) fields
    predicted_fraud=int(model.predict(single_row)[0]),
    llm_explainer=MyExplainer(),
)
```

Whatever `explain` returns is returned by `explain_with_llm`, and errors it raises are passed on to the caller. The explainer is passed per call rather than stored on the interpreter, so a fitted `ModelInterpreter` stays picklable.

### Registering a default explainer

A package can make its explainer the default, so `llm_explainer` can be left out, by registering a factory function under the `model_interpreter.llm_explainers` entry point group in its `pyproject.toml`:

```toml
[project.entry-points."model_interpreter.llm_explainers"]
default = "my_package:get_explainer"
```

`get_explainer()` takes no arguments and returns the explainer. When `explain_with_llm` is called without `llm_explainer`, it loads and calls the factory registered as `default`. If none is installed, it raises an `ImportError` asking for an explainer to be passed.

## Installation

The easiest way to get `model_interpreter` is directly from [pypi](https://pypi.org/project/model_interpreter/) with;

 `pip install model_interpreter`


## Examples

To help get started there are example notebooks in the [examples](https://github.com/lvgig/model_interpreter/tree/main/examples) folder in the repo that show how to use each transformer.

To open the example notebooks in [binder](https://mybinder.org/) click [here](https://mybinder.org/v2/gh/lvgig/model_interpreter/HEAD?labpath=examples) or click on the `launch binder` shield above and then click on the directory button in the side bar to the left to navigate to the specific notebook.

## Issues

For bugs and feature requests please open an [issue](https://github.com/lvgig/model_interpreter/issues).

## Build and test

The test framework we are using for this project is [pytest](https://docs.pytest.org/en/stable/). To build the package locally and run the tests follow the steps below.

First clone the repo and move to the root directory;

```shell
git clone https://github.com/lvgig/model_interpreter.git
cd model_interpreter
```

Next install `model_interpreter` and development dependencies;

```shell
pip install . -r requirements-dev.txt
```

Finally run the test suite with `pytest`;

```shell
pytest
```

## Contribute

`model_interpreter` is under active development, we're super excited if you're interested in contributing! 
