PYTHON3 := $(shell which python3 2>/dev/null)

PYTHON := python3

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  format [check=1]   to apply black formatter; use with 'check=1' to check instead of modify (requires black)"
	@echo "  lint               to run pylint on source files"

.PHONY : clean
clean:
	rm -rf src/qml_benchmarks.egg-info/
	rm -rf src/qml_benchmarks/__pycache__/
	rm -rf src/qml_benchmarks/models/__pycache__/

.PHONY:format
format:
ifdef check
	black -l 100 ./src/qml_benchmarks --check
else
	black -l 100 ./src/qml_benchmarks
endif

.PHONY: lint
lint:
	pylint src/qml_benchmarks --rcfile .pylintrc

