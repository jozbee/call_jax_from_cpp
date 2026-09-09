/* Preloaded allocation counter, used to find out who allocates in the call
 * path.
 *
 * Grepping the source for `malloc` proves nothing about what the linked binary
 * does at run time. This library interposes the allocator and counts calls
 * between an arm and a disarm marker, so a test can wrap exactly the region it
 * cares about.
 *
 * Build:  cc -shared -fPIC -O2 -o malloc_guard.so malloc_guard.c -ldl
 * Use:    LD_PRELOAD=./malloc_guard.so ./driver ...            (ELF)
 *         DYLD_INSERT_LIBRARIES=./malloc_guard.so ./driver ...  (Mach-O)
 *
 * The two platforms interpose differently and both halves are here. On ELF,
 * defining `malloc` in a preloaded object shadows libc's, and the real one is
 * reached through dlsym(RTLD_NEXT); dlsym may itself allocate while
 * resolving, which would recurse forever, so allocations made during that
 * window come from a static arena, recognised by address and skipped on free.
 * On Mach-O the two-level namespace makes shadowing by name a no-op, so
 * dyld's __DATA,__interpose table is used instead; dyld does not apply it to
 * the image that declares it, so the replacements call malloc/free directly.
 *
 * A whole-process zero is unachievable -- XLA's thunk runtime allocates
 * thousands of times per call inside the plugin -- so every armed allocation
 * is attributed to the module containing its return address:
 *
 *   CLS_SELF     the main executable, or a module matching $PJRT_GUARD_SELF
 *                (default "libpjrt_exec") -- the number that must stay at zero
 *   CLS_PLUGIN   inside libpjrt_c_api_cpu_plugin
 *   CLS_RUNTIME  libc, libstdc++, LAPACK, the thread pool, everything else
 *
 * The C++ operator new family is interposed too, under its mangled names, so
 * that an inlined std::vector growth is charged to the module that grew the
 * vector rather than to libstdc++. operator delete is left alone: libstdc++
 * routes it to free, which is already interposed.
 *
 * Classification is ELF-only. On Apple the counts are still correct;
 * pjrt_guard_classified() reports 0 and the class counters stay 0.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#define PJRT_GUARD_CLASSIFY 0
#else
#define PJRT_GUARD_CLASSIFY 1
#include <link.h>
#endif

#define CLS_SELF 0
#define CLS_PLUGIN 1
#define CLS_RUNTIME 2
#define CLS_COUNT 3

static int g_armed;
static unsigned long g_allocs;
static unsigned long g_frees;
static unsigned long g_class[CLS_COUNT];

/* Counted whether armed or not. Thousands in total with none while armed
 * means the interposer is live and the path is clean; zero in total means the
 * preload never took effect. */
static unsigned long g_total_allocs;

/* --- caller classification (ELF) ----------------------------------------- */

#if PJRT_GUARD_CLASSIFY

/* One executable segment of one loaded object. Segments of distinct objects do
 * not overlap, so a sorted table can be binary-searched. */
struct guard_range {
  uintptr_t lo;
  uintptr_t hi;
  int cls;
};

/* Overflow is not an error: the surplus ranges classify as CLS_RUNTIME. */
#define GUARD_MAX_RANGES 512

static struct guard_range g_ranges[GUARD_MAX_RANGES];

/* Published only once the table is complete, so a racing allocation
 * classifies as CLS_RUNTIME rather than indexing a half-written entry. */
static size_t g_nranges;

static int guard_range_class(const char *name) {
  const char *self;
  if (name == NULL || name[0] == '\0') {
    return CLS_SELF; /* an empty dlpi_name is the main executable */
  }
  if (strstr(name, "pjrt_c_api_cpu_plugin") != NULL) {
    return CLS_PLUGIN;
  }
  self = getenv("PJRT_GUARD_SELF");
  if (self == NULL || self[0] == '\0') {
    self = "libpjrt_exec";
  }
  if (strstr(name, self) != NULL) {
    return CLS_SELF;
  }
  return CLS_RUNTIME;
}

static int guard_phdr_cb(struct dl_phdr_info *info, size_t size, void *data) {
  size_t *n = (size_t *)data;
  int cls;
  size_t i;
  (void)size;
  cls = guard_range_class(info->dlpi_name);
  for (i = 0; i < info->dlpi_phnum; ++i) {
    const ElfW(Phdr) *ph = &info->dlpi_phdr[i];
    /* Only executable segments can hold a return address. */
    if (ph->p_type != PT_LOAD || (ph->p_flags & PF_X) == 0) {
      continue;
    }
    if (*n >= GUARD_MAX_RANGES) {
      return 1; /* table full: stop walking */
    }
    g_ranges[*n].lo = (uintptr_t)info->dlpi_addr + (uintptr_t)ph->p_vaddr;
    g_ranges[*n].hi = g_ranges[*n].lo + (uintptr_t)ph->p_memsz;
    g_ranges[*n].cls = cls;
    ++*n;
  }
  return 0;
}

/* Called from pjrt_guard_arm(), never from an interposed function:
 * dl_iterate_phdr takes the loader lock, and the timed region must not. */
static void guard_build_ranges(void) {
  size_t n = 0;
  size_t i;
  __atomic_store_n(&g_nranges, (size_t)0, __ATOMIC_RELEASE);
  dl_iterate_phdr(guard_phdr_cb, &n);
  for (i = 1; i < n; ++i) {
    struct guard_range key = g_ranges[i];
    size_t j = i;
    while (j > 0 && g_ranges[j - 1].lo > key.lo) {
      g_ranges[j] = g_ranges[j - 1];
      --j;
    }
    g_ranges[j] = key;
  }
  __atomic_store_n(&g_nranges, n, __ATOMIC_RELEASE);
}

static int guard_classify(const void *ra) {
  uintptr_t a = (uintptr_t)ra;
  size_t lo = 0;
  size_t hi = __atomic_load_n(&g_nranges, __ATOMIC_ACQUIRE);
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (a < g_ranges[mid].lo) {
      hi = mid;
    } else if (a >= g_ranges[mid].hi) {
      lo = mid + 1;
    } else {
      return g_ranges[mid].cls;
    }
  }
  /* JIT-emitted code, a vDSO frame, or an object mapped after arming. */
  return CLS_RUNTIME;
}

#endif /* PJRT_GUARD_CLASSIFY */

static void count_alloc(const void *ra) {
  (void)ra;
  __atomic_add_fetch(&g_total_allocs, 1, __ATOMIC_RELAXED);
  if (!__atomic_load_n(&g_armed, __ATOMIC_RELAXED)) {
    return;
  }
  __atomic_add_fetch(&g_allocs, 1, __ATOMIC_RELAXED);
#if PJRT_GUARD_CLASSIFY
  __atomic_add_fetch(&g_class[guard_classify(ra)], 1, __ATOMIC_RELAXED);
#endif
}

static void count_free(void) {
  if (__atomic_load_n(&g_armed, __ATOMIC_RELAXED)) {
    __atomic_add_fetch(&g_frees, 1, __ATOMIC_RELAXED);
  }
}

#if defined(__APPLE__)

/* dyld rebinds every other image's references to `replacee` so they land on
 * `replacement`; this image keeps the originals. */
#define PJRT_INTERPOSE(replacement, replacee)                            \
  __attribute__((used, section("__DATA,__interpose"))) static struct {      \
    const void *replacement;                                                \
    const void *replacee;                                                   \
  } pjrt_interpose_##replacee = {(const void *)(unsigned long)&replacement, \
                                    (const void *)(unsigned long)&replacee}

static void *guard_malloc(size_t n) {
  const void *ra = __builtin_return_address(0);
  count_alloc(ra);
  return malloc(n);
}

static void *guard_calloc(size_t nmemb, size_t size) {
  const void *ra = __builtin_return_address(0);
  count_alloc(ra);
  return calloc(nmemb, size);
}

static void *guard_realloc(void *p, size_t n) {
  const void *ra = __builtin_return_address(0);
  count_alloc(ra);
  return realloc(p, n);
}

static void guard_free(void *p) {
  count_free();
  free(p);
}

static int guard_posix_memalign(void **out, size_t align, size_t n) {
  const void *ra = __builtin_return_address(0);
  count_alloc(ra);
  return posix_memalign(out, align, n);
}

static void *guard_aligned_alloc(size_t align, size_t n) {
  const void *ra = __builtin_return_address(0);
  count_alloc(ra);
  return aligned_alloc(align, n);
}

static void *guard_valloc(size_t n) {
  const void *ra = __builtin_return_address(0);
  count_alloc(ra);
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
  const void *ra = __builtin_return_address(0);
  if (resolving) return boot_alloc(n);
  if (!real_malloc) resolve();
  count_alloc(ra);
  return real_malloc(n);
}

void *calloc(size_t nmemb, size_t size) {
  const void *ra = __builtin_return_address(0);
  if (resolving) {
    void *p = boot_alloc(nmemb * size);
    if (p) memset(p, 0, nmemb * size);
    return p;
  }
  if (!real_calloc) resolve();
  count_alloc(ra);
  return real_calloc(nmemb, size);
}

void *realloc(void *p, size_t n) {
  const void *ra = __builtin_return_address(0);
  if (!real_realloc) resolve();
  if (from_boot(p)) {
    /* Grew a bootstrap block: hand back a real one. The old size is unknown,
     * so copy up to the end of the arena -- always a safe upper bound. */
    void *q = real_malloc(n);
    if (q) {
      size_t avail = (size_t)(boot_arena + sizeof boot_arena - (char *)p);
      memcpy(q, p, n < avail ? n : avail);
    }
    count_alloc(ra);
    return q;
  }
  count_alloc(ra);
  return real_realloc(p, n);
}

void free(void *p) {
  if (from_boot(p)) return;
  if (!real_free) resolve();
  count_free();
  real_free(p);
}

int posix_memalign(void **out, size_t align, size_t n) {
  const void *ra = __builtin_return_address(0);
  if (!real_posix_memalign) resolve();
  count_alloc(ra);
  return real_posix_memalign(out, align, n);
}

void *aligned_alloc(size_t align, size_t n) {
  const void *ra = __builtin_return_address(0);
  if (!real_aligned_alloc) resolve();
  count_alloc(ra);
  return real_aligned_alloc(align, n);
}

/* --- operator new family ------------------------------------------------- *
 * These call real_malloc directly, so one `new` counts once. The mangled names
 * are the same on x86_64 and aarch64 (LP64: size_t mangles to `m`); nothrow_t
 * and align_val_t are declared as plain words, since only the slots matter. */

/* operator new(0) must still return a distinct, freeable pointer. */
static void *guard_new(size_t n, const void *ra) {
  if (resolving) return boot_alloc(n ? n : 1);
  if (!real_malloc) resolve();
  count_alloc(ra);
  return real_malloc(n ? n : 1);
}

static void *guard_new_aligned(size_t n, size_t align, const void *ra) {
  void *p = NULL;
  if (!real_posix_memalign) resolve();
  count_alloc(ra);
  if (n == 0) n = 1;
  /* posix_memalign rejects alignments below sizeof(void *); over-aligning
   * keeps the memory freeable with plain free(). */
  if (align < sizeof(void *)) align = sizeof(void *);
  if (real_posix_memalign(&p, align, n) != 0) return NULL;
  return p;
}

/* The throwing forms must not return null, and C cannot throw bad_alloc. */
static void *or_abort(void *p) {
  if (p == NULL) abort();
  return p;
}

/* operator new(size_t) */
void *_Znwm(size_t n) {
  return or_abort(guard_new(n, __builtin_return_address(0)));
}

/* operator new[](size_t) */
void *_Znam(size_t n) {
  return or_abort(guard_new(n, __builtin_return_address(0)));
}

/* operator new(size_t, const std::nothrow_t &) */
void *_ZnwmRKSt9nothrow_t(size_t n, const void *tag) {
  (void)tag;
  return guard_new(n, __builtin_return_address(0));
}

/* operator new[](size_t, const std::nothrow_t &) */
void *_ZnamRKSt9nothrow_t(size_t n, const void *tag) {
  (void)tag;
  return guard_new(n, __builtin_return_address(0));
}

/* operator new(size_t, std::align_val_t) */
void *_ZnwmSt11align_val_t(size_t n, size_t align) {
  return or_abort(guard_new_aligned(n, align, __builtin_return_address(0)));
}

/* operator new[](size_t, std::align_val_t) */
void *_ZnamSt11align_val_t(size_t n, size_t align) {
  return or_abort(guard_new_aligned(n, align, __builtin_return_address(0)));
}

/* operator new(size_t, std::align_val_t, const std::nothrow_t &) */
void *_ZnwmSt11align_val_tRKSt9nothrow_t(size_t n, size_t align,
                                         const void *tag) {
  (void)tag;
  return guard_new_aligned(n, align, __builtin_return_address(0));
}

/* operator new[](size_t, std::align_val_t, const std::nothrow_t &) */
void *_ZnamSt11align_val_tRKSt9nothrow_t(size_t n, size_t align,
                                         const void *tag) {
  (void)tag;
  return guard_new_aligned(n, align, __builtin_return_address(0));
}

#endif

/* --- markers, resolved by the program under test via dlsym --------------- */

void pjrt_guard_arm(void) {
#if PJRT_GUARD_CLASSIFY
  /* Before arming, so a freshly dlopened plugin is in the table before its
   * allocations arrive. */
  guard_build_ranges();
#endif
  g_allocs = 0;
  g_frees = 0;
  memset(g_class, 0, sizeof g_class);
  __atomic_store_n(&g_armed, 1, __ATOMIC_RELAXED);
}

void pjrt_guard_disarm(void) {
  __atomic_store_n(&g_armed, 0, __ATOMIC_RELAXED);
}

unsigned long pjrt_guard_alloc_count(void) { return g_allocs; }

unsigned long pjrt_guard_free_count(void) { return g_frees; }

unsigned long pjrt_guard_total_alloc_count(void) { return g_total_allocs; }

/* 0 for an unknown class, or everywhere when classification is unavailable. */
unsigned long pjrt_guard_alloc_count_class(int cls) {
  if (cls < 0 || cls >= CLS_COUNT) return 0;
  return g_class[cls];
}

/* False on Apple, and before the first arm(), which builds the ranges. */
int pjrt_guard_classified(void) {
#if PJRT_GUARD_CLASSIFY
  return __atomic_load_n(&g_nranges, __ATOMIC_ACQUIRE) > 0;
#else
  return 0;
#endif
}

/* Tells "zero allocations" from "the preload silently did nothing". */
int pjrt_guard_present(void) { return 1; }
