// Exact (unsampled) libc heap-in-use peak tracker, for validating the
// mallinfo2 sampler in scripts/benchmark/v2/instrument.py.
//
// Interposes malloc/calloc/realloc/free and maintains a live byte counter via
// malloc_usable_size(), so no bookkeeping map is needed (and hence no
// allocation recursion). Exposes hp_reset_peak()/hp_get_peak() for per-stage
// windows, callable from Python through ctypes.
#define _GNU_SOURCE
#include <stdlib.h>
#include <malloc.h>
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdint.h>

static void *(*real_malloc)(size_t);
static void *(*real_calloc)(size_t, size_t);
static void *(*real_realloc)(void *, size_t);
static void  (*real_free)(void *);

static atomic_llong live = 0;
static atomic_llong peak = 0;
static __thread int in_hook = 0;

// Bootstrap buffer: dlsym() itself may calloc before real_calloc is resolved.
static char boot[65536];
static size_t boot_off = 0;

static void init_syms(void) {
    static int done = 0;
    if (done) return;
    done = 1;
    real_malloc  = dlsym(RTLD_NEXT, "malloc");
    real_calloc  = dlsym(RTLD_NEXT, "calloc");
    real_realloc = dlsym(RTLD_NEXT, "realloc");
    real_free    = dlsym(RTLD_NEXT, "free");
}

static void bump(long long delta) {
    long long now = atomic_fetch_add(&live, delta) + delta;
    long long p = atomic_load(&peak);
    while (now > p && !atomic_compare_exchange_weak(&peak, &p, now)) { }
}

void hp_reset_peak(void) { atomic_store(&peak, atomic_load(&live)); }
long long hp_get_peak(void)  { return atomic_load(&peak); }
long long hp_get_live(void)  { return atomic_load(&live); }

void *malloc(size_t n) {
    if (!real_malloc) init_syms();
    void *p = real_malloc(n);
    if (p && !in_hook) { in_hook = 1; bump((long long)malloc_usable_size(p)); in_hook = 0; }
    return p;
}

void *calloc(size_t a, size_t b) {
    if (!real_calloc) {
        if (!real_malloc) {
            size_t need = a * b;
            if (boot_off + need > sizeof boot) return NULL;
            void *p = boot + boot_off; boot_off += need; return p;
        }
        init_syms();
    }
    void *p = real_calloc(a, b);
    if (p && !in_hook) { in_hook = 1; bump((long long)malloc_usable_size(p)); in_hook = 0; }
    return p;
}

void *realloc(void *old, size_t n) {
    if (!real_realloc) init_syms();
    long long before = old ? (long long)malloc_usable_size(old) : 0;
    void *p = real_realloc(old, n);
    if (!in_hook) {
        in_hook = 1;
        bump((p ? (long long)malloc_usable_size(p) : 0) - before);
        in_hook = 0;
    }
    return p;
}

void free(void *p) {
    if (p >= (void *)boot && p < (void *)(boot + sizeof boot)) return;  // bootstrap block
    if (!real_free) init_syms();
    if (p && !in_hook) { in_hook = 1; bump(-(long long)malloc_usable_size(p)); in_hook = 0; }
    real_free(p);
}
