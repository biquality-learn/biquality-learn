.PHONY: quality format test test-examples coverage docs

# Check that source code meets quality standards

quality:
	black --check examples bqlearn
	flake8 bqlearn examples
	isort --check-only bqlearn examples

# Format source code automatically

format:
	isort examples bqlearn
	black examples bqlearn

# Run tests for the library

test:
	pytest -n auto -s -v bqlearn

# Run tests for examples

test-examples:
	pytest -n auto -s -v examples

# Run code coverage

coverage:
	pytest -n auto --cov --cov-report xml -s -v bqlearn

# Check that docs can build

docs:
	cd docs && make clean && make html