# Haze Log

Context that used to live as inline comments in the source, moved here when the
tier-placement pass brought these files under `comment-guard` (CLAUDE.md: no comments
in code; a doc-comment above a declaration is the only exception).

## Domain errors and tokens have one home each

Domain errors live in `app/src/errors/<domain>.py` and enums in
`app/src/tokens/<domain>.py`, not in per-domain `errors.py` / `enums.py` modules beside
each `core.py`. This is CLAUDE.md's Organization taxonomy, enforced by the `code-tiers`
semgrep rules `error-outside-errors-py` and `enum-outside-tokens-py`. haze sits outside
moon's project graph, but the `gates` CI job builds its file list from the git diff and
never consults that graph, so the rules apply here regardless.

## Open design notes

- **End token.** The end token is currently added as a motor onto the last layer's
  decoders by `Haze.set_lexical_chain` when the network is sequential. It may be better
  for the decoder to carry the end token by default, leaving `set_lexical_chain` to only
  mark it active when a run needs it.

## Known defects, recorded rather than fixed

These predate the tier-placement pass and were deliberately left alone by it, since that
pass was a placement change that had to preserve behaviour exactly.

- `app/src/decoder/models.py` imports `Series` and `Sequential` from
  `app/src/decoder/core.py`, which defines neither, so that module cannot be imported.
- `InvalidRewardError.__init__` calls `super().__init__(self, ...)`, so the exception's
  `args` carry the exception instance ahead of the message.
- `Connector.set_strength` calls `Registry.set_strength`, which `Registry` does not
  define.
- `Motor.reset_state` assigns `self._state = 0`, but `Motor.get_state` sums
  `self._signals`, which the reset leaves untouched.
- The test suite is red (17 failed, 18 passed, 5 errors). Nothing in the workspace runs
  it, because haze is parked out of moon's project graph.
