# Packaging

How this repo declares its dependencies and how it is installed and tested.

## Domain Description

Haze ships as a single Python distribution named `haze`, built with setuptools from
`pyproject.toml`. Before this, dependencies were listed in a bare `requirements.txt`
holding two lines (`numpy`, `pytest`) and there was no build manifest at all, so the
repo could not be installed as a package, could not be built into a wheel, and was
invisible to moon's Python toolchain — the toolchain keys on `pyproject.toml`, so the
unit inherited neither a build task nor a test task and its suite was unreachable from
`moon run :test`.

## Policies

### The distribution is built from `pyproject.toml`, never `requirements.txt`

Given a fresh checkout, when someone installs the repo, then `python -m pip install -e
'.[dev]'` is the whole install: `numpy` is the one runtime dependency and `pytest` is the
one development dependency, and both are declared in `pyproject.toml`.

`requirements.txt` is removed. It duplicated the same two names in a format no build
backend reads, and the packaging workflow's PyPI path only consulted it behind an
`if [ -f requirements.txt ]` guard, so its absence changes nothing there while
`python -m build` now has the manifest it always required.

### The import root is the repository root, and `app` is the package tree

Given a test importing `app.src.haze.core`, when the distribution is installed, then that
import resolves, because `app` and every directory beneath it are PEP 420 namespace
packages discovered by `packages.find` with `namespaces = true`.

`app/tests` is excluded from the distribution: it is the suite that proves the package,
not part of it. This mapping records the layout that exists rather than improving it. A
published library whose top-level import is `app` is not the shape haze should ship in,
and the `feat-169` branch already carries the repackaging that renames the tree to a
single `haze` root. Rewriting the imports of 50 modules here would collide with that work
head-on, so the branch that owns the rename keeps it.

### The version is a floor, not a claim of continuity

Given `feat-169` declares `0.2.0`, when this branch declares `0.1.0`, then the two do not
compete for the same immutable PyPI version. Nothing on this branch has been published;
`0.2.0` belongs to the repackaged tree on `feat-169`, and reusing it here would mean two
different source trees claiming one version number.

### Tests run under pytest from the repository root

Given `python -m pytest`, when it runs from the repo root, then `testpaths` selects
`app/tests` and `pythonpath = ["."]` puts the repo root on `sys.path`, so the suite's
`app.src.*` imports resolve whether or not the distribution is installed.

The suite is currently red on its own — 27 failed, 8 passed, 5 errors — and this
packaging change neither caused that nor repairs it. It makes the failures reachable from
`moon run haze:test`, where before they were invisible.

### The moon tasks override the inherited Python defaults

Given `.moon/tasks/python.yml` compiles `src` and discovers `unittest` tests under `src`,
when haze keeps its sources in `app`, then haze overrides `build`, `test` and `install` in
its own `moon.yml` with `mergeArgs: 'replace'`.

Adding `pyproject.toml` is what admits the unit to the Python toolchain; it is not enough
on its own, because the shared base tasks assume a `src/` layout this repo does not have.
