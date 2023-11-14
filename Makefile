.PHONY: code-style type-check pre-test tests


code-style:
	flake8 bin src tests
	black --check bin src tests

type-check:
	mypy bin src tests --namespace-packages

pre-test: code-style type-check

tests:
	pytest tests/
