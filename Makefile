.PHONY: code-style type-check pre-test


code-style:
	flake8 src
	black --check src

type-check:
	mypy src --namespace-packages

pre-test: code-style type-check
