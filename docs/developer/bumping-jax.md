# Bumping JAX

Moving to a new JAX release means moving the XLA commit under it, rebasing the
fork's patches, rebuilding the plugin, re-exporting every artifact, and only
then publishing a plugin release. `versions.env` is edited first and everything
else follows from it — nothing in this repository restates a version by hand.

This page is written to be executed, by a person or an agent, in order. Every
step has the same three parts: what to do, one command that says whether it
worked, and where to look when it did not. Skipping a verify line is how a bump
gets to step 14 before anyone finds out that step 5 was wrong.

The current pin is JAX {{ jax_version }} on XLA `{{ xla_commit }}`, fork branch
`{{ xla_fork_branch }}` at `{{ xla_fork_commit }}`, PJRT C API
{{ pjrt_api_major }}.{{ pjrt_api_minor }}. What that bump actually found is
recorded at the bottom as a worked example.

## Step 0 — Prerequisites and budget

**Do.** Set aside most of a day, of which the bazel plugin build is 30–60
minutes per architecture on a real machine with a cold cache — and hours if it
is cross-built under QEMU. You need `bazel` (bazelisk is fine), `docker`, `gh`
authenticated with push rights to the fork, `xxd`, `liblapack-dev` and
`libblas-dev`, roughly 25 GB of free disk, and the `third_party/xla` submodule
checked out. **Never run the plugin build concurrently with a benchmark**: it
saturates every core, and a latency number measured beside it is invalid rather
than noisy ([measurement](measurement.md)).

**Verify.** `for t in bazel docker gh xxd; do command -v $t || echo "MISSING $t"; done`
→ expected: four paths and no `MISSING` line.

**If it fails.** `gh auth status` is the one that bites late rather than early:
it is not needed until step 9, by which point the day's work is uncommitted.
Fix it now.

## Step 1 — Choose the version and read the JAX changelog

**Do.** Pick the JAX release. Read its changelog specifically for changes to
export and AOT — `jax.export`, `serialize_executable`, the StableHLO
compatibility window, and anything about the CPU backend's custom calls. Set
`JAX_VERSION` and `JAXLIB_VERSION` in `versions.env`. They must match exactly:
jaxlib is where the compiled artefacts of a JAX release live, and a mismatched
pair produces failures that look like compiler bugs.

**Verify.** `grep -E '^(JAX|JAXLIB)_VERSION' versions.env`
→ expected: two lines carrying the identical version.

**If it fails.** jaxlib sometimes appears on PyPI hours after jax. If only one
of the pair exists yet, stop and wait; do not pin a jaxlib from the previous
release.

## Step 2 — Find the XLA commit JAX pins

**Do.** JAX names its XLA revision in `third_party/xla/revision.bzl` at the
release tag. Read it at `jax-v<version>` and write it into `XLA_COMMIT` in
`versions.env`.

**Verify.**
`curl -s https://raw.githubusercontent.com/jax-ml/jax/jax-v<version>/third_party/xla/revision.bzl | grep -o '[0-9a-f]\{40\}'`
→ expected: exactly the hash now in `XLA_COMMIT`.

**If it fails.** A 404 means the tag does not exist yet — releases appear on
PyPI before the tag sometimes. If the file has moved, look for the `http_archive`
rule that fetches XLA in the workspace files; the commit is in its URL.

## Step 3 — Branch the fork from that commit

**Do.** In `third_party/xla`, fetch the upstream commit and cut a new branch
from it: `cjfc/jax-v<version>`. **Never rewrite a published branch.** The old
branch stays exactly where it is, because a checkout somewhere else names it,
and rewriting history under a submodule pointer produces the same "commit not
found" failure as never pushing at all.

**Verify.**
`git -C third_party/xla merge-base --is-ancestor $XLA_COMMIT HEAD && echo ok`
→ expected: `ok`.

**If it fails.** The fork remote may not have the upstream commit yet. Add
`https://github.com/openxla/xla.git` as a second remote, fetch it, then branch;
pushing the branch to the fork carries the objects along.

## Step 4 — Cherry-pick patch 1 and re-apply the BUILD link edge

**Do.** Cherry-pick the LAPACK-kernels commit onto the new branch. The one
conflict to expect is `xla/pjrt/c/BUILD`, which upstream edits often. Re-apply
by hand what the patch adds to the `pjrt_c_api_cpu_plugin` target: a dependency
on `//jaxlib_cpu_kernels`, the four Debian multiarch `-L` paths, and the
`-llapack -lblas` link options.

**Verify.**
`grep -nE 'jaxlib_cpu_kernels|^\s*"-L|-llapack' third_party/xla/xla/pjrt/c/BUILD`
→ expected: the dependency line, exactly four `-L` entries, all of them
multiarch LAPACK or BLAS directories, `-llapack` and `-lblas`, and no conflict
markers anywhere in the file. Check the `-L` list even when the cherry-pick
reports no conflict: this hunk auto-merges cleanly and losing it fails at the
final link, after the whole build, with a message that reads like a missing
system package. Four is also a ceiling, not just a floor — the comment above
the list says why a fifth, more general path breaks the binary.

**If it fails.** If the plugin target has been renamed or split upstream,
find the rule that produces `pjrt_c_api_cpu_plugin.so` and add the edge there;
the patch file under `third_party/patches/` shows exactly which attributes it
touched.

## Step 5 — Re-vendor the jaxlib CPU kernels from the new tag

**Do.** Copy from `jax-ml/jax` at `jax-v<version>`:
`jaxlib/cpu/lapack_kernels.cc`, `jaxlib/cpu/lapack_kernels.h`,
`jaxlib/cpu/cpu_kernels.cc` and `jaxlib/cpu/ffi_helpers.h`, into
`jaxlib_cpu_kernels/` in the fork.

**Diff the new copies against the ones already vendored before you take them.**
The tree currently carries the `jax-v0.11.1` copies against a `jax-v0.11.0`
pin, deliberately, and re-vendoring from the pinned tag without reading
[the fork page](xla-fork.md) would silently undo that. The rule is not "always
take the pinned tag"; it is "take the pinned tag unless a specific, recorded
reason says otherwise, and record the reason where the next person will look". Rewrite the includes: `jaxlib/cpu/...`
becomes the local path, while `xla/ffi/api/...` stays as it is. **Drop the
sparse and tridiagonal handler registrations** — their kernels are not vendored
here, so registering them does not link. Add whatever new dependencies the
sources have acquired to the package's BUILD file.

**Verify.**
`grep -nE '^\s*JAX_CPU_REGISTER_HANDLER.*(sparse|tridiagonal)' third_party/xla/jaxlib_cpu_kernels/cpu_kernels.cc`
→ expected: no output. Match the registrations, not the word: the file carries
a comment naming both dropped handlers, so a bare `grep sparse` reports a
failure on a correct tree.

**If it fails.** A registration whose kernel is missing fails at link with an
undefined symbol naming the handler. An executable that needs one of the
dropped handlers fails at load with `No FFI handler registered for
cpu_csr_sparse_dense_ffi on a platform Host`, which is the intended, legible
outcome rather than a crash.

## Step 6 — Re-create patch 2

**Do.** Apply the create-options patch to
`xla/pjrt/c/pjrt_c_api_cpu_internal.cc`. Check first what upstream now parses
itself and do not duplicate it — `asynchronous` moved upstream at this pin, and
parsing it twice is a conflict rather than a redundancy. What the patch must
keep is `PJRT_Plugin_Attributes_Cpu`, returning `GetXlaPluginCAttributes()` plus
the `int64` markers `supports_synchronous_execution`,
`supports_max_inflight_computations` and `cjfc_plugin_patch_level`, wired into
`CreatePjrtApi`. Bump `cjfc_plugin_patch_level` if the surface changes shape.

**Verify.**
`grep -c 'supports_synchronous_execution' third_party/xla/xla/pjrt/c/pjrt_c_api_cpu_internal.cc`
→ expected: at least 1.

**If it fails.** The usual cause is that the attributes plumbing changed
upstream. Start from `GetXlaPluginCAttributes()` and follow how the generic
`PJRT_Plugin_Attributes` is assembled; the patch only prepends entries to that
vector.

## Step 7 — Build and smoke-test the plugin

**Do.** `make plugin-source`, or
`docker compose -f docker/compose.yml run --rm plugin-builder tools/build_plugin.sh`
when bazel is not on this machine. Then build the probe and point it at the
result: the probe is the only thing that reports what the plugin actually
supports rather than what it was meant to.

**Verify.**
`make tools && PJRT_CPU_PLUGIN=build/plugin/libpjrt_c_api_cpu_plugin.so build/bin/plugin_probe`
→ expected: it prints the plugin's API version and an attribute list containing
`supports_synchronous_execution`.

**If it fails.** A missing `GetPjrtApi` export means bazel built something other
than the plugin. A link error naming `dgetrf_`/`dgesvd_` means `liblapack-dev`
is absent or patch 1's link edge did not survive step 4. `xxd: command not
found` is XLA's own build rules, not ours.

## Step 8 — Re-vendor the PJRT C headers

**Do.** Copy `xla/pjrt/c/pjrt_c_api.h` and `xla/pjrt/c/pjrt_c_api_cpu.h` from
the fork into `third_party/pjrt/`, update `third_party/pjrt/VERSION`, and set
`PJRT_API_MAJOR`/`PJRT_API_MINOR` in `versions.env`. Then fix whatever the
header change breaks. Every `Args` struct in this tree is value-initialised and
then assigned field by field, deliberately: a struct that gains a trailing field
then gets a zero for it instead of failing to compile, and designated
initialisers would turn every header bump into a mechanical edit across the
codebase. Keep it that way.

**Verify.**
`grep -E '^#define PJRT_API_MINOR' third_party/pjrt/pjrt_c_api.h`
→ expected: the same number as `PJRT_API_MINOR` in `versions.env`.

**If it fails.** `make lib` is the fuller check. A `struct_size` mismatch at run
time — rather than a compile error — means the vendored header and the built
plugin disagree; they must come from the same fork commit.

## Step 9 — Push the fork branch, then regenerate the patches

**Do.** Push the branch. **Ask the user before pushing**, then push, then
regenerate the patch files:

```
git -C third_party/xla push origin <branch>
git -C third_party/xla format-patch "$XLA_COMMIT..HEAD" \
    -o third_party/patches --no-signature
```

and write the resulting `HEAD` hash into `XLA_FORK_COMMIT`. A patch must exist
in three places — a pushed branch, a regenerated patch file, and the hash in
`versions.env` — before anything is allowed to depend on it. See
[the lost-commit lesson](xla-fork.md#the-lost-commit-lesson).

**Verify.**
`test "$(git -C third_party/xla ls-remote origin <branch> | cut -f1)" = "$(git -C third_party/xla rev-parse HEAD)" && echo pushed`
→ expected: `pushed`.

**If it fails.** If `ls-remote` prints nothing, the branch does not exist on the
remote and the local commit is unreachable by anybody, including you after the
next `git checkout`. This is the failure this project has already had once.

## Step 10 — Update the submodule pointer and versions.env

**Do.** Set the branch in `.gitmodules`, move the submodule pointer to the
pushed commit, and make sure `XLA_FORK_BRANCH` and `XLA_FORK_COMMIT` in
`versions.env` name the same thing.

**Verify.** `git submodule status third_party/xla`
→ expected: the hash printed with no leading `+` or `-`, and equal to
`XLA_FORK_COMMIT`.

**If it fails.** A leading `+` means the submodule's checked-out commit differs
from what the superproject records; a `-` means it is not initialised. Neither
is a state to commit.

## Step 11 — Pin jax and jaxlib, and lock

**Do.** Update the `jax==` and `jaxlib==` pins in `pyproject.toml` to match
`versions.env`, then `uv lock` and `uv sync`.

**Verify.**
`uv run python -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"`
→ expected: the new version printed twice.

**If it fails.** `tests/python/test_versions.py` is the standing check that
`pyproject.toml` and `versions.env` agree; run it before wondering which file is
wrong.

(ifrt-envelope)=
## Step 12 — Verify the exporter's private-API path

**Do.** The exporter reaches into JAX internals to serialize a compiled
executable, and that is the part of this project most likely to break on a bump.
Export the smallest example and let the checker read it back.

**Verify.** `build/bin/plugin_probe --view` -> expected: `view_supported=1`,
`view_created=1`, `view_aliases=1`, `view_sees_later_write=1`. This one is
checked because it has already changed: the CPU plugin was once documented as
not implementing it at all. A change here does not break anything, but it
invalidates a paragraph in {doc}`runtime-internals`.

**Verify.** `make export && uv run python -m jax2exec check artifacts/basic`
→ expected: the sidecar's signature is printed and the artifact is reported as
runnable on this host.

**If it fails.** Look for a renamed private module or a changed signature on the
serialize call, and fix it in `python/jax2exec/export.py` rather than pinning
around it. If `jax.export` has grown a supported public route to the same
bytes, prefer it and note the change in the changelog.

### The envelope check, which is the part that fails quietly

This step has a second half that is easy to skip and expensive to miss.

jaxlib does not hand out PJRT bytes. Since its client moved onto IFRT, both
`client.serialize_executable(loaded)` and `loaded.serialize()` return the same
thing: an IFRT envelope wrapping the payload the PJRT C API actually wants.
Handing that to `PJRT_Executable_DeserializeAndLoad` fails with

```
PjRtCpuClient::DeserializeExecutable proto deserialization failed
```

The layout is `varint(header_length) || header_proto || pjrt_payload`, where the
header names the format (`pjrt_ifrt`) and carries device and sharding
information, and the payload is byte-for-byte the `ExecutableAndOptionsProto`.
`jax2exec._ifrt` strips it, recognising the envelope by walking the
header's protobuf fields rather than matching bytes, and passing anything it
does not recognise through untouched.

What makes this worth a checklist entry is the failure mode. Nothing crashes.
The loader falls back to compiling the `.mlirbc`, so the answers stay correct
and the tests stay green — you simply lose ahead-of-time loading, and pay a
compile at every startup, without being told.

**Verify.**

```console
$ uv run python -c "import json;print(json.load(open('artifacts/basic.json'))['artifacts']['executable_source'])"
ifrt-unwrapped
$ build/bin/example_01_basic | grep load_kind
load_kind=deserialized
```

→ expected: an `executable_source` of `ifrt-unwrapped` or `as-is`, and a
`load_kind` of **`deserialized`**. A `load_kind` of `compiled` here means the
envelope changed shape and the unwrap no longer recognises it.

**If it fails.** Dump the first bytes of the artifact and compare them with the
format above. A payload begins with tag `0x0a`, field 1 of
`ExecutableAndOptionsProto`. Teach `_ifrt.py` the new shape; do not disable the
check, and do not accept the compile fallback as normal — it is the symptom.

## Step 13 — Re-export every artifact on the target machine

**Do.** `make export`, and regenerate the reference cases, **on the machine that
will execute them**. A `.binpb` embeds machine code for the exporting host; it
is relinked at load, never recompiled.

**Verify.** `uv run python -m jax2exec check artifacts/trajopt`
→ expected: the recorded `isa_level` is reported as satisfied by this host.

**If it fails.** An artifact exported on a newer machine fails as an illegal
instruction with a backtrace that says nothing about artifacts, which is exactly
what `FunctionOptions::isa_guard` exists to convert into a `LoadError`. Re-export
here, or run from the `.mlirbc` with `LoadPolicy::CompileOnly`.

## Step 14 — Run the full verification gate on an idle machine

**Do.** In order: `make test`, then `make test-alloc`, then `make test-slow`,
and `make test-rt` if the host is tuned for it. Then compare the new pin against
the old one with `tools/run_matrix.sh`, interleaved — never sequentially.
Correctness first: a latency number from a run that computed the wrong answer is
not a conservative estimate, it is noise.

**Check what you are actually testing against first.** A plugin built from the
previous pin loads and runs artifacts exported by the new one, correctly and
without complaint, so `build/plugin/` holding a stale binary is invisible to
every test in the tree. Nothing detects it. Confirm by hand:

```console
$ grep -E '^(jax_version|fork_commit)' build/plugin/PLUGIN_INFO.txt
$ grep -E '^(JAX_VERSION|XLA_FORK_COMMIT)' versions.env
```

→ expected: the two `jax_version`/`JAX_VERSION` lines agree, and so do the two
commits. `make plugin` after a `PLUGIN_RELEASE` bump, or `make plugin-source`,
is what puts the right one there.

**Verify.** `cut -d' ' -f1 /proc/loadavg && make test`
→ expected: a load average below about 0.5 before the run, and pytest exiting 0.

**If it fails.** A failing allocation gate points at the wrapper, not at XLA:
the count that must be zero is ours. A tail that has regressed against the
previous pin is worth a bisect through the fork's commits before it is blamed on
the JAX release.

## Step 15 — Update the docs and open one PR

**Do.** Most pages need no edit, because every version they mention is a
substitution out of `versions.env`. Update what is genuinely prose: the worked
example at the bottom of this page, [the fork page](xla-fork.md) if the patches
changed shape, the compatibility notes, and `CHANGELOG.md`. Then open **one**
pull request containing the files that must move together, because a subset of
them is a broken tree:

- `versions.env`, `.gitmodules`, the `third_party/xla` submodule pointer, and
  `docker/Dockerfile`'s `XLA_REF`, which is the one place the fork branch is
  hand-typed and the one nothing tests;
- `third_party/patches/*.patch` and `third_party/pjrt/*`;
- `pyproject.toml` and `uv.lock`;
- the docs and changelog edits.

**Verify.** `make docs`
→ expected: sphinx exits 0. The build runs with `-W`, so a stale
cross-reference fails it.

**If it fails.** `--keep-going` is already on, so one build reports every
warning; fix them all rather than one per run.

## Step 16 — Last, publish the plugin release

**Do.** Only now, when the fork branch is pushed and the tree is green, publish
the plugin:

```
tools/release_plugin.sh --version <version> --update-manifest
```

Review the `tools/plugin_versions.txt` diff it prints, commit it, and set
`PLUGIN_RELEASE` in `versions.env` to the new tag. Both go into the same pull
request as step 15, before it merges: `PLUGIN_RELEASE` must never name a tag
that does not exist yet on `main`, because `make plugin` is the first thing a
new user runs.

**Verify.** In a fresh clone of the branch: `make plugin`
→ expected: `get_plugin: sha256 ok` followed by
`get_plugin: installed .../libpjrt_c_api_cpu_plugin.so`.

**If it fails.** A missing asset for one architecture is a partial release, not
a broken one: `tools/get_plugin.sh` prints the remedies, and the other
architecture can be added to the same tag later. A checksum mismatch means the
manifest row and the published asset disagree, which is a stop-everything
condition — re-run with `--update-manifest` and commit the corrected row.

## What the 0.9.0.1 to 0.11.1 bump found

Recorded because it is the worked example, and because most of it will be true
again next time.

**The PJRT C API moved from minor 90 to 114**, and among the structs this code
uses only two changed: `PJRT_ExecuteOptions` grew four trailing fields, and
`PJRT_Executable_DeserializeAndLoad_Args` gained `load_options`. Both were
absorbed with no edit at all, because every `Args` struct here is
value-initialised and then assigned rather than built with designated
initialisers. That is the whole reason for the convention.

**The LAPACK kernels changed substantially**: `lapack_kernels.cc` grew from 1656
to 2644 lines between the two JAX releases, and the new sources needed
dependencies the old package did not have — `absl/functional:function_ref`,
`absl/synchronization`, `absl/status` and `@tsl//tsl/platform`. Carrying the old
copies forward was not an option, since the FFI handler signatures have to match
what the pinned release lowers to.

**Upstream absorbed half of patch 2.** XLA now parses the `asynchronous` create
option itself, so the patch shrank to `max_inflight_computations` plus the
attributes. In the same change upstream started **validating option names and
rejecting unknown ones** with `InvalidArgument("Unexpected option name passed to
PJRT_Client_Create")`. That reversed a rule this project had relied on — unknown
options used to be ignored silently — and is why `Runtime` now queries
`PJRT_Plugin_Attributes` before creating a client and retries without a rejected
option.

**The build stayed in WORKSPACE mode.** XLA's `.bazelversion` is 8.7.0 and its
own `.bazelrc` sets `--noenable_bzlmod --enable_workspace`, so nothing here
passes `--config=bzlmod`. Do not add it on the assumption that a newer bazel
requires it.

## What the 0.11.1 to 0.11.0 move found

The second worked example, and the first one that ran the procedure *downward*.
The reason for moving back is recorded in
[open threads](open-threads.md); what follows is only what the mechanics
turned up.

**A downward move is every step above, unchanged.** Nothing about the procedure
assumes the new version is newer. The XLA revision, the fork branch, the
patches, the lock file and the artifacts all move the same way. Do not
shortcut it on the theory that the destination is a place the tree has been
before — it has not been. This repository went 0.9.0.1 to 0.11.1 directly, so
0.11.0 was a new base with no branch, no release and no patch set to reuse.

**The PJRT C API did not move, and that collapsed step 8 to one line.**
`xla/pjrt/c/pjrt_c_api.h` is byte-identical between the two XLA revisions, both
0.114, and `pjrt_c_api_cpu.h` did not change either. Only the `xla_commit=`
line in `third_party/pjrt/VERSION` needed editing. Check this before budgeting
for step 8; between adjacent releases it is often free.

**Step 4's expected conflict did not happen, and step 6's did.** `xla/pjrt/c/BUILD`
auto-merged cleanly, which is the outcome that most deserves distrust — run the
verify line anyway and confirm both the `//jaxlib_cpu_kernels` dependency and
the `-llapack -lblas` link options survived, because a clean merge that dropped
one of them fails much later, at link. The create-options patch conflicted
instead, and in the direction a downward move creates: the 0.11.1 base already
parsed upstream's `process_id` option, the patch sat immediately beside it, and
against a base without `process_id` the cherry-pick offered upstream's code and
ours as one block.

**Resolve that conflict to the patch's purpose, not to what compiles.**
`CpuClientOptions` carries a `process_id` field at both revisions, so keeping
upstream's parsing would have built fine. It was dropped regardless. The patch
exists to accept `max_inflight_computations` and to advertise what the plugin
supports; carrying a backport of an unrelated upstream option inside it widens
the fork's surface for no caller. `cjfc_plugin_patch_level` stayed at 2,
because the three markers describe what *this patch* adds and all three still
hold.

The stock surface underneath the patch did move, and it is worth being exact
about it, because the markers exist to make exactly this discoverable and here
they cannot. `ValidateCreateOptions` rejects an unlisted option name, so a
caller that sends `process_id` succeeds against the published 0.11.1 plugin and
fails `PJRT_Client_Create` against this one, while both report
`cjfc_plugin_patch_level = 2`. Nothing in this project sends it — `Runtime`
sends `cpu_device_count`, `asynchronous` and `max_inflight_computations`, and
`RuntimeOptions` has no escape hatch for an arbitrary option — so the marker
was left alone rather than spent on a distinction no caller here can observe.
A third party driving the plugin directly would not be able to tell the two
builds apart from the attributes, which is a real if narrow gap.

**`format-patch` does not remove the files it replaces.** Patch 2's subject
changed, so its filename changed, and the old file stayed on disk beside the
new one. Delete `third_party/patches/000*.patch` before regenerating, or the
tree carries a stale patch that still applies and no longer matches the branch.

**A plugin loaded an artifact exported against a different XLA revision.** That
was not known before. Take it as one observation across two adjacent revisions,
not as a guarantee: there is no version field on the CPU deserialize path at
either commit, so nothing *enforces* a match, but nothing promises one either,
and the two revisions pin different LLVM commits. It is cheap to re-check and
worth re-checking on each bump. The published 0.11.1
plugin loaded a 0.11.0 export with `executable_source` of `ifrt-unwrapped` and
a `load_kind` of `deserialized`, and returned correct results, on both example
fixtures. Two things follow. The step 12 envelope check can be run before the
new plugin exists, which moves the most bump-fragile step much earlier. And an
A/B across two JAX releases can hold the plugin fixed and vary only the
artifact, which isolates export-time code generation exactly — the comparison
[open threads](open-threads.md) asks for before this pin moves forward again.

**A `-L` that fixes the link can silently ruin the binary.** The build fails on
a non-Debian host with `cannot find -llapack`, because the patch only knows
about Debian's multiarch directories. Adding `/usr/lib` to the list in the fork
makes it link, and produces a plugin that needs `GLIBC_2.44` where the
published one needs `GLIBC_2.27`, with `librt`, `libdl` and `libpthread` gone
from `DT_NEEDED`. A `-L` is not scoped to the library you added it for: it is
searched ahead of the hermetic sysroot for every implicit `-l` too. Nothing in
the build warns, the plugin loads and runs perfectly on the machine that made
it, and the damage is visible only in `objdump -T | grep GLIBC`. Pass the path
with `--arch-flags` for a local build and leave the tree alone; the flag lands
in `PLUGIN_INFO.txt` where it marks the artifact unpublishable.

**Check `glibc_max` before publishing anything, and compare it with the asset
you are replacing.** `PLUGIN_INFO.txt` records it, which is what made the
regression above findable at all. Equal or lower than the previous release is
the only acceptable answer.

**The `--out` path is resolved against the XLA tree, not the repository.**
`tools/build_plugin.sh` `cd`s into `$XLA_DIR`, so `--out build/plugin` writes
`third_party/xla/build/plugin` and leaves the submodule dirty, which then fails
step 10's verify for a reason that has nothing to do with the patches. Pass an
absolute path.

**The lock file moved two packages and no others.** `uv lock` re-resolved
`jax` and `jaxlib` — versions and wheel digests, 21 lines added and 28 removed
— and left every transitive dependency alone: `ml_dtypes`, `numpy` and `scipy`
are untouched. A bump that churns the transitive set is telling you something;
this one did not, which is part of why it was cheap.

**Going backward can cost you an interpreter, though.** jaxlib 0.11.1 publishes
cp312, cp313, cp314 and cp315 wheels; 0.11.0 publishes the first three and no
cp315. `requires-python` in `pyproject.toml` is `>=3.12` with no ceiling, so a
contributor on Python 3.15 who could resolve this tree yesterday cannot today,
and `uv` reports it as an unsatisfiable resolution rather than as a pin
problem. `mise.toml` pins 3.12.14, so nothing here noticed. A downward bump
should check the wheel matrix of the release it is moving to before assuming
the environment is the easy part.
