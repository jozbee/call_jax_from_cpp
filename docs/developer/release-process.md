# Release process

There are two kinds of release here and they are deliberately independent. A
**source release** is a tagged tree: the C++ library, the exporter and the docs.
A **plugin release** is a prebuilt PJRT CPU plugin for one JAX version,
published as GitHub Release assets, because there is no official prebuilt CPU
PJRT C-API plugin anywhere — jaxlib links its CPU client statically and never
exports `GetPjrtApi`.

They move on different clocks. A source release happens when this project
changes; a plugin release happens when the JAX pin moves.

## The version, and where it lives

One number covers both halves of the project, and it lives in `pyproject.toml`
as the `jax2exec` distribution version. `docs/conf.py` reads it back through
`importlib.metadata`, so the site's version is the installed package's version
and cannot drift from it. There is no version macro in the C++ headers: the
library is vendored into a consumer's build rather than installed as a
versioned artifact, and a second number would be a second thing to forget.

Every *pinned* version — JAX, jaxlib, the XLA commit, the fork branch and
commit, the PJRT API version, the plugin release tag — lives in `versions.env`
instead, and is available in any page here as a MyST substitution. Never
hand-type one.

Source releases are tagged `v<version>`. Plugin releases are tagged
`plugin-jax-v<jax version>`, are published with `make_latest: false`, and are
not source releases: "latest" has to keep meaning the latest release of this
project.

## The plugin release convention

The names are load-bearing. `tools/plugin_versions.txt`,
`cmake/GetPjrtPlugin.cmake` and `make plugin` all resolve an asset by name, so
the two producers must agree exactly:

```
tag     plugin-jax-v<jax version>
assets  pjrt_cpu_plugin-jax-v<jax version>-linux-<arch>.tar.gz
        pjrt_cpu_plugin-jax-v<jax version>-linux-<arch>.tar.gz.sha256
inside  libpjrt_c_api_cpu_plugin.so, PLUGIN_INFO.txt, LICENSE-xla
```

`PLUGIN_INFO.txt` is what makes an asset auditable a year later: the JAX
version, the XLA commit, the fork commit and branch, the PJRT API minor, the
build mode, the host architecture, the maximum glibc symbol version the binary
needs, and its runtime dependencies. The plugin binary is Apache-2.0 from XLA
and ships `LICENSE-xla` inside the tarball, even though this repository is
released under the Unlicense.

Assets are built **baseline** for their architecture — no `--config=avx_*`, no
`ARCH_FLAGS` — so one asset runs on any machine of that kind. The ISA-locked
half of this system is the serialized executable, not the plugin.

### The two producers must emit identical assets

- `tools/release_plugin.sh` on a developer machine, through
  `docker buildx --target plugin-export`. This is the primary producer: it has
  real cores and a warm bazel cache.
- `.github/workflows/plugin.yml`, triggered by a `plugin-jax-v*` tag or run
  manually. This is the reproducible secondary path — it proves the tarball can
  be regenerated from nothing but `versions.env` and the fork, and it fills in an
  architecture the developer machine does not have.

The workflow's honest cost is hours: a from-scratch XLA CPU plugin build
compiles LLVM from source, and a hosted runner has four vCPUs. Its
`timeout-minutes` is 355 against a hard limit of 360, because a job killed at
the limit tells you nothing about why it was slow.

Either producer may publish either architecture into the same tag. Cross-building
one architecture on the other under QEMU works and takes hours; building each one
natively and uploading both to the same tag is much faster.

`release_plugin.sh` refuses to replace an asset that already exists unless
`--force` is passed, and the refusal is the point: replacing a binary under a
name whose sha256 is already recorded breaks every checkout that pinned it.

### The rule about PLUGIN_RELEASE

**`PLUGIN_RELEASE` must point at a published tag before the pull request that
sets it merges.** `make plugin` is the first command a new user runs, and a
release tag that does not exist yet turns that into a download failure on a
clean clone. Publish, verify by downloading the asset back, commit the manifest
rows, then merge. [Bumping JAX](bumping-jax.md) has this as its final step for
exactly this reason.

Until an asset exists for a platform, `tools/get_plugin.sh` prints the three
remedies — build from the fork, publish the asset, or point `PJRT_CPU_PLUGIN` at
a plugin you already have — and CI treats a missing asset as a missing artifact
rather than a broken tree.

## The docs deploy

`.github/workflows/docs.yml` builds the site on every push to `main` and
deploys it to GitHub Pages. The build runs with `-W --keep-going`: a warning is
a failure, and one run reports all of them. `ci.yml` runs the same build on
every pull request without the deploy, so a broken cross-reference fails the PR
rather than the deploy.

**One manual step, once per repository:** Settings → Pages → Source: "GitHub
Actions". Without it the deploy job fails with a 404 from the Pages API.
`.nojekyll` is written by `sphinx.ext.githubpages`, which is what keeps
`_static/` and `_sources/` from 404ing.

## CI action pins

The three workflows pin every action they use, and the pins were chosen for
one reason: Node 20 leaves GitHub's runners in September 2026, and an action
still on a Node 20 major stops working there. Verified 2026-09-05 with
`gh api repos/<owner>/<repo>/releases/latest --jq .tag_name` for the latest
release, then each `action.yml` read for its runtime.

| Action | Latest then | Pinned | Runtime |
|---|---|---|---|
| `actions/checkout` | v7.0.1 | v7 | node24 |
| `actions/cache` | v6.1.0 | v6 | node24 |
| `actions/upload-artifact` | v7.0.1 | v7 | node24 |
| `actions/download-artifact` | v8.0.1 | v8 | node24 |
| `astral-sh/setup-uv` | v10.0.1 | v10.0.1 | node24 |
| `docker/setup-buildx-action` | v4.3.0 | v4 | node24 |
| `docker/build-push-action` | v7.3.0 | v7 | node24 |
| `actions/configure-pages` | v6.0.0 | v6 | node24 |
| `actions/upload-pages-artifact` | v5.0.0 | v5 | composite |
| `actions/deploy-pages` | v5.0.1 | v5 | node24 |
| `bazel-contrib/setup-bazel` | 0.19.0 | 0.19.0 | node24 |
| `softprops/action-gh-release` | v3.0.3 | v3 | node24 |

Two are pinned to an exact release rather than a major alias:
`astral-sh/setup-uv` stopped publishing a floating major tag after v7, and
`bazel-contrib/setup-bazel` publishes none (its releases are 0.x). To re-check
the table, run the two commands above for each row; a runtime other than
`node24` or `composite` is the thing to act on.

## Release checklist

Source release, in order. Each line has the command that says whether it
worked.

1. **The tree is green on an idle machine.**
   `cut -d' ' -f1 /proc/loadavg && make test && make test-alloc`
   → pytest exits 0 and the allocation census reports zero wrapper allocations
   per call.
2. **The docs build clean.** `make docs`
   → sphinx exits 0 under `-W`.
3. **External links still resolve.** `make docs-linkcheck`
   → no broken links. Run it before a release even though CI does not: an
   upstream page that moved is not worth failing every PR over, and is worth
   fixing once here.
4. **The version is bumped.** `grep '^version' pyproject.toml`
   → the new version, and `uv sync` has been re-run so the installed
   distribution matches.
5. **The changelog says what changed.** `sed -n '1,40p' CHANGELOG.md`
   → an entry for this version, with anything removed listed under Removed.
   Keep a Changelog format, semantic versioning.
6. **The pinned versions agree with themselves.** `uv run pytest tests/python/test_versions.py`
   → passes; `pyproject.toml` and `versions.env` are consistent.
7. **The plugin release for the pinned JAX version exists.**
   `make plugin` in a fresh clone
   → `get_plugin: sha256 ok`. If this is a JAX bump, see
   [bumping JAX](bumping-jax.md) step 16 first.
8. **Tag and push.** `git tag v<version> && git push origin v<version>`
   → the docs workflow republishes the site from `main`; the plugin workflow is
   not triggered by this tag and should not be.

A release that fails step 7 is not a release: the tree builds, and then nothing
runs.
