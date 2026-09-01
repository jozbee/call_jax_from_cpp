/* Preloaded allocation counter, used to prove the call path never allocates.
 *
 * Grepping the source for `malloc` proves nothing about what the linked binary
 * does at run time -- OpenBLAS, libm, and the C++ runtime are all free to
 * allocate behind the kernel's back. This library interposes the allocator and
 * counts calls between an arm and a disarm marker, so a test can wrap exactly
 * the region it cares about (the steady-state calls, after warm-up).
 *
 * Build:  cc -shared -fPIC -O2 -o malloc_guard.so malloc_guard.c -ldl
 * Use:    LD_PRELOAD=./malloc_guard.so ./driver ...            (ELF)
 *         DYLD_INSERT_LIBRARIES=./malloc_guard.so ./driver ...  (Mach-O)
 *
 * The two platforms interpose differently and both halves are here:
 *
 *   ELF     defining `malloc` in a preloaded object shadows libc's, and the
 *           real one is reached through dlsym(RTLD_NEXT). dlsym may itself
 *           allocate while resolving, which would recurse forever, so
 *           allocations made during that window come from a static arena and
 *           are recognised (by address) and ignored on free.
 *
 *   Mach-O  the two-level namespace makes shadowing by name a no-op, so dyld's
 *           __DATA,__interpose table is used instead. dyld does not apply the
 *           interposition to the image that declares it, so the replacements
 *           call malloc/free directly and no bootstrap arena is needed.
 *
 * The program under test does not link against this library. It looks the
 * markers up with dlsym(RTLD_DEFAULT, ...) and skips them when absent, so the
 * same binary runs with and without the preload.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_armed;
static unsigned long g_allocs;
static unsigned long g_frees;

/* Counted whether armed or not, from the first relocation onwards. Its only
 * job is to make a zero armed count believable: if the process made thousands
 * of allocations in total and none while armed, the interposer is clearly
 * live and the call path is clearly clean. A zero here would instead mean the
 * preload never took effect. */
static unsigned long g_total_allocs;

static void count_alloc(void) {
  __atomic_add_fetch(&g_total_allocs, 1, __ATOMIC_RELAXED);
  if (g_armed) __atomic_add_fetch(&g_allocs, 1, __ATOMIC_RELAXED);
}

static void count_free(void) {
  if (g_armed) __atomic_add_fetch(&g_frees, 1, __ATOMIC_RELAXED);
}

#if defined(__APPLE__)

/* dyld rebinds every other image's references to `replacee` so they land on
 * `replacement`; this image keeps the originals, which is what lets the
 * replacements below call malloc and free by name. */
#define PJRT_INTERPOSE(replacement, replacee)                            \
  __attribute__((used, section("__DATA,__interpose"))) static struct {      \
    const void *replacement;                                                \
    const void *replacee;                                                   \
  } pjrt_interpose_##replacee = {(const void *)(unsigned long)&replacement, \
                                    (const void *)(unsigned long)&replacee}

static void *guard_malloc(size_t n) {
  count_alloc();
  return malloc(n);
}

static void *guard_calloc(size_t nmemb, size_t size) {
  count_alloc();
  return calloc(nmemb, size);
}

static void *guard_realloc(void *p, size_t n) {
  count_alloc();
  return realloc(p, n);
}

static void guard_free(void *p) {
  count_free();
  free(p);
}

static int guard_posix_memalign(void **out, size_t align, size_t n) {
  count_alloc();
  return posix_memalign(out, align, n);
}

static void *guard_aligned_alloc(size_t align, size_t n) {
  count_alloc();
  return aligned_alloc(align, n);
}

static void *guard_valloc(size_t n) {
  count_alloc();
  return valloc(n);
}

PJRT_INTERPOSE(guard_malloc, malloc);
PJRT_INTERPOSE(guard_calloc, calloc);
PJRT_INTERPOSE(guard_realloc, realloc);
PJRT_INTERPOSE(guard_free, free);
PJRT_INTERPOSE(guard_posix_memalign, posix_memalign);
PJRT_INTERPOSE(guard_aligned_alloc, aligned_alloc);
PJRT_INTERPOSE(guard_valloc, valloc);

#else /* ELF: shadow the symbols and reach the real ones through RTLD_NEXT. */

static void *(*real_malloc)(size_t);
static void *(*real_calloc)(size_t, size_t);
static void *(*real_realloc)(void *, size_t);
static void (*real_free)(void *);
static int (*real_posix_memalign)(void **, size_t, size_t);
static void *(*real_aligned_alloc)(size_t, size_t);

/* Served to whatever allocates while we are resolving the real allocator. */
static char boot_arena[1 << 16];
static size_t boot_used;
static int resolving;

static int from_boot(const void *p) {
  return (const char *)p >= boot_arena &&
         (const char *)p < boot_arena + sizeof boot_arena;
}

static void *boot_alloc(size_t n) {
  n = (n + 15u) & ~(size_t)15u;
  if (boot_used + n > sizeof boot_arena) return NULL;
  void *p = boot_arena + boot_used;
  boot_used += n;
  return p;
}

static void resolve(void) {
  if (resolving) return;
  resolving = 1;
  real_malloc = (void *(*)(size_t))dlsym(RTLD_NEXT, "malloc");
  real_calloc = (void *(*)(size_t, size_t))dlsym(RTLD_NEXT, "calloc");
  real_realloc = (void *(*)(void *, size_t))dlsym(RTLD_NEXT, "realloc");
  real_free = (void (*)(void *))dlsym(RTLD_NEXT, "free");
  real_posix_memalign =
      (int (*)(void **, size_t, size_t))dlsym(RTLD_NEXT, "posix_memalign");
  real_aligned_alloc =
      (void *(*)(size_t, size_t))dlsym(RTLD_NEXT, "aligned_alloc");
  resolving = 0;
}

void *malloc(size_t n) {
  if (resolving) return boot_alloc(n);
  if (!real_malloc) resolve();
  count_alloc();
  return real_malloc(n);
}

void *calloc(size_t nmemb, size_t size) {
  if (resolving) {
    void *p = boot_alloc(nmemb * size);
    if (p) memset(p, 0, nmemb * size);
    return p;
  }
  if (!real_calloc) resolve();
  count_alloc();
  return real_calloc(nmemb, size);
}

void *realloc(void *p, size_t n) {
  if (!real_realloc) resolve();
  if (from_boot(p)) {
    /* Grew a bootstrap block: hand back a real one. The old size is unknown,
     * so copy up to the end of the arena -- always a safe upper bound. */
    void *q = real_malloc(n);
    if (q) {
      size_t avail = (size_t)(boot_arena + sizeof boot_arena - (char *)p);
      memcpy(q, p, n < avail ? n : avail);
    }
    count_alloc();
    return q;
  }
  count_alloc();
  return real_realloc(p, n);
}

void free(void *p) {
  if (from_boot(p)) return;
  if (!real_free) resolve();
  count_free();
  real_free(p);
}

int posix_memalign(void **out, size_t align, size_t n) {
  if (!real_posix_memalign) resolve();
  count_alloc();
  return real_posix_memalign(out, align, n);
}

void *aligned_alloc(size_t align, size_t n) {
  if (!real_aligned_alloc) resolve();
  count_alloc();
  return real_aligned_alloc(align, n);
}

#endif

/* --- markers, resolved by the program under test via dlsym --------------- */

void pjrt_guard_arm(void) {
  g_allocs = 0;
  g_frees = 0;
  __atomic_store_n(&g_armed, 1, __ATOMIC_RELAXED);
}

void pjrt_guard_disarm(void) {
  __atomic_store_n(&g_armed, 0, __ATOMIC_RELAXED);
}

unsigned long pjrt_guard_alloc_count(void) { return g_allocs; }

unsigned long pjrt_guard_free_count(void) { return g_frees; }

unsigned long pjrt_guard_total_alloc_count(void) { return g_total_allocs; }

/* Proves the interposer is actually live, so a test can tell "zero allocations"
 * apart from "the preload silently did nothing". */
int pjrt_guard_present(void) { return 1; }
