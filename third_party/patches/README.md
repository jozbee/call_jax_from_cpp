# XLA fork patches

The three commits this project carries on top of the XLA revision that the
pinned JAX release uses (`XLA_COMMIT` in `versions.env`), regenerated with

    git -C third_party/xla format-patch "$XLA_COMMIT..HEAD" \
        -o third_party/patches --no-signature

They exist so the fork can be reconstructed from this repository alone.

A patch has to live in three places before anything is allowed to depend on
it, and a rebase is not finished until all three agree:

1. a branch pushed to the fork, referenced by commit hash;
2. these files, regenerated from that branch;
3. `XLA_FORK_COMMIT` in `versions.env`.

That rule is not bureaucracy. The previous synchronous-execution patch was
committed locally and named by the submodule pointer, but never pushed, so the
commit could not be fetched by anybody — including its author — and had to be
written again from the prose description that survived in the notes.

See `docs/developer/xla-fork.md` for what each patch does and
`docs/developer/bumping-jax.md` for the procedure that regenerates them.
