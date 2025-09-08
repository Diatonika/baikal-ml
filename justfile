set dotenv-load := true
set windows-shell := ["pwsh.exe", "-NoLogo", "-Command"]

# Directories for Linter & Formatter

source-dirs := "src tests notebooks"
nexus-read-user := env("NEXUS_READ_USER")
nexus-read-pass := env("NEXUS_READ_PASS")

default:
    @just --list

[group("setup")]
init *install-options: && (install install-options)
    poetry config virtualenvs.in-project true --local
    poetry config installer.re-resolve false --local
    poetry config http-basic.baikal-pypi {{ nexus-read-user }} {{ nexus-read-pass }} --local

[group("setup")]
init-dev: (init "--extras" "cpu")

[group("setup")]
init-lint: (init "--only" "main,lint,test" "--extras" "cpu")

[group("setup")]
init-test: (init "--only" "main,test" "--extras" "cpu")

[group("misc")]
install *options:
    poetry install {{ options }}

[group("misc")]
update *options:
    poetry update {{ options }}

[group("lint")]
lint-check: init-lint
    poetry run -- ruff check {{ source-dirs }} --config pyproject.toml
    poetry run -- dmypy run --export-types -- {{ source-dirs }} --config-file pyproject.toml --cache-fine-grained

[group("lint")]
lint-fix: init-lint
    poetry run ruff check --fix-only {{ source-dirs }}

[group("format")]
format-check: init-lint
    poetry run ruff format --diff {{ source-dirs }}

[group("format")]
format-fix: init-lint
    poetry run ruff format {{ source-dirs }}

[group("test")]
test-all: init-test
    poetry run pytest

[group("notebooks")]
run-jupyter: init-dev
    jupyter notebook