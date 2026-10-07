// ztrace: every kernel launch of a process, through CUPTI's API callbacks (CUDA_INJECTION64_PATH=ztrace.so): the
// kernel's name, grid, block, dynamic shared memory, the PDL attribute and each parameter's bytes (cuFuncGetParamInfo),
// one JSON line a distinct launch, with the phase zrec.py sets (ztrace_phase). An aligned 8-byte word that is a CUDA
// address (cuPointerGetAttribute) is written as <ptr> and left out of the launch's identity, so launches that differ
// only by their buffers collapse into one line. A launch through the runtime is taken at the driver call beneath it
// when CUPTI reports that call, else at the runtime call ("via":"rt"). For the Zig port's conformance tests of the
// kernels that are not Triton's (the extensions' and torch's).
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cupti.h>      // with generated_cuda_runtime_api_meta.h (the runtime launch parameters)
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE* out;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static __thread char phase[96] = "other";
static __thread int rt_in, rt_seen;     // inside a runtime launch call; its driver launch was seen
static uint64_t seen[1 << 20];          // hashes of the distinct launches written (open addressing)
static uint64_t ptr_v[1 << 16];         // cuPointerGetAttribute answers by value (direct mapped)
static unsigned char ptr_p[1 << 16];
static unsigned long long launches, written, by_rt, unresolved;
typedef int (*func_by_symbol_t)(void**, const void*);
static func_by_symbol_t func_by_symbol;

void ztrace_phase(const char* p) {
    strncpy(phase, p ? p : "other", sizeof(phase) - 1);
    phase[sizeof(phase) - 1] = 0;
}

static uint64_t fnv(uint64_t h, const void* p, size_t n) {
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

static int first_time(uint64_t h) {
    if (!h) h = 1;
    for (size_t i = h & ((1 << 20) - 1), k = 0; k < (1 << 20); i = (i + 1) & ((1 << 20) - 1), ++k) {
        if (seen[i] == h) return 0;
        if (!seen[i]) { seen[i] = h; return 1; }
    }
    return 0;
}

// A CUDA address (the driver's answer, kept by value; device memory sits both low, 0x3..., and high, 0xf8..., in the
// 48-bit user range). Called under mu.
static int is_ptr(uint64_t v) {
    if (v < 0x10000ull || (v >> 48) != 0) return 0;
    size_t i = (size_t)((v * 0x9E3779B97F4A7C15ull) >> 48);
    if (ptr_v[i] == v) return ptr_p[i];
    unsigned int mt = 0;
    int p = cuPointerGetAttribute(&mt, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)v) == CUDA_SUCCESS;
    ptr_v[i] = v;
    ptr_p[i] = (unsigned char)p;
    return p;
}

static void put_json_str(FILE* f, const char* s) {
    for (; *s; ++s) {
        if (*s == '"' || *s == '\\') fputc('\\', f);
        if ((unsigned char)*s >= 0x20) fputc(*s, f);
    }
}

// The CUfunction that answers name and parameter queries: a runtime launch hands the driver a CUkernel (a library's
// kernel, for any context) where a CUfunction goes, and its function in the current context is the one to ask.
static CUfunction resolve(CUfunction f, const char** name) {
    const char* n = NULL;
    if (!f) return NULL;
    if (cuFuncGetName(&n, f) == CUDA_SUCCESS) {
        if (!*name) *name = n;
        return f;
    }
    CUfunction g = NULL;
    if (cuKernelGetFunction(&g, (CUkernel)f) == CUDA_SUCCESS && g && cuFuncGetName(&n, g) == CUDA_SUCCESS) {
        if (!*name) *name = n;
        return g;
    }
    if (!*name && cuKernelGetName(&n, (CUkernel)f) == CUDA_SUCCESS) *name = n;
    return NULL;
}

static void record(const char* name, CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                   unsigned bz, unsigned smem, int pdl, void** params, const char* via) {
    pthread_mutex_lock(&mu);
    ++launches;
    if (via[0] == 'r') ++by_rt;
    f = resolve(f, &name);
    if (!name) name = "?";
    static char hex[16384];
    size_t hp = 0;
    hex[0] = 0;
    uint64_t h = fnv(1469598103934665603ull, phase, strlen(phase));
    h = fnv(h, name, strlen(name));
    unsigned dims[7] = {gx, gy, gz, bx, by, bz, smem};
    h = fnv(h, dims, sizeof(dims));
    h = fnv(h, &pdl, sizeof(pdl));
    if (!f || !params) {
        ++unresolved;
        hp += snprintf(hex + hp, sizeof(hex) - hp, "?");
    }
    for (int i = 0; f && params && i < 256; ++i) {
        size_t off = 0, sz = 0;
        if (cuFuncGetParamInfo(f, (size_t)i, &off, &sz) != CUDA_SUCCESS) break;
        const unsigned char* b = (const unsigned char*)params[i];
        if (hp + 2 * sz + 32 >= sizeof(hex)) { hp += snprintf(hex + hp, sizeof(hex) - hp, "|..."); break; }
        hp += snprintf(hex + hp, sizeof(hex) - hp, "%s%zu:", i ? "|" : "", off);
        for (size_t j = 0; j < sz;) {
            if (sz - j >= 8 && (j % 8) == 0) {
                uint64_t v;
                memcpy(&v, b + j, 8);
                if (is_ptr(v)) {
                    hp += snprintf(hex + hp, sizeof(hex) - hp, "<ptr>");
                    h = fnv(h, "P", 1);
                    j += 8;
                    continue;
                }
            }
            hp += snprintf(hex + hp, sizeof(hex) - hp, "%02x", b[j]);
            h = fnv(h, b + j, 1);
            ++j;
        }
        h = fnv(h, &sz, sizeof(sz));
    }
    if (first_time(h)) {
        ++written;
        fputs("{\"phase\":\"", out);
        put_json_str(out, phase);
        fputs("\",\"name\":\"", out);
        put_json_str(out, name);
        fprintf(out, "\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"smem\":%u,\"pdl\":%d,\"via\":\"%s\",\"params\":\"%s\"}\n",
                gx, gy, gz, bx, by, bz, smem, pdl, via, hex);
    }
    pthread_mutex_unlock(&mu);
}

static int rt_pdl(const cudaLaunchConfig_t* c) {
    for (unsigned i = 0; c && i < c->numAttrs; ++i)
        if (c->attrs[i].id == cudaLaunchAttributeProgrammaticStreamSerialization)
            return c->attrs[i].val.programmaticStreamSerializationAllowed;
    return 0;
}

static CUfunction rt_func(const void* sym) {
    void* f = NULL;
    if (func_by_symbol && sym && func_by_symbol(&f, sym) == 0) return (CUfunction)f;
    return NULL;
}

static void CUPTIAPI callback(void* ud, CUpti_CallbackDomain dom, CUpti_CallbackId id, const void* data) {
    (void)ud;
    const CUpti_CallbackData* d = (const CUpti_CallbackData*)data;
    if (dom == CUPTI_CB_DOMAIN_DRIVER_API) {
        if (d->callbackSite != CUPTI_API_ENTER) return;
        if (rt_in) rt_seen = 1;
        if (id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel || id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz) {
            const cuLaunchKernel_params* p = (const cuLaunchKernel_params*)d->functionParams;
            record(d->symbolName, p->f, p->gridDimX, p->gridDimY, p->gridDimZ, p->blockDimX, p->blockDimY, p->blockDimZ,
                   p->sharedMemBytes, 0, p->kernelParams, "drv");
        } else {
            const cuLaunchKernelEx_params* p = (const cuLaunchKernelEx_params*)d->functionParams;
            const CUlaunchConfig* c = p->config;
            int pdl = 0;
            for (unsigned i = 0; i < c->numAttrs; ++i)
                if (c->attrs[i].id == CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION)
                    pdl = c->attrs[i].value.programmaticStreamSerializationAllowed;
            record(d->symbolName, p->f, c->gridDimX, c->gridDimY, c->gridDimZ, c->blockDimX, c->blockDimY, c->blockDimZ,
                   c->sharedMemBytes, pdl, p->kernelParams, "drv");
        }
        return;
    }
    if (dom != CUPTI_CB_DOMAIN_RUNTIME_API) return;
    if (d->callbackSite == CUPTI_API_ENTER) {
        rt_in = 1;
        rt_seen = 0;
        return;
    }
    rt_in = 0;
    if (rt_seen) return;
    // CUPTI did not report a driver launch beneath this runtime one: take it from the runtime call
    if (id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 || id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000) {
        const cudaLaunchKernel_v7000_params* p = (const cudaLaunchKernel_v7000_params*)d->functionParams;
        record(d->symbolName, rt_func(p->func), p->gridDim.x, p->gridDim.y, p->gridDim.z, p->blockDim.x, p->blockDim.y,
               p->blockDim.z, (unsigned)p->sharedMem, 0, p->args, "rt");
    } else {
        const cudaLaunchKernelExC_v11060_params* p = (const cudaLaunchKernelExC_v11060_params*)d->functionParams;
        const cudaLaunchConfig_t* c = p->config;
        record(d->symbolName, rt_func(p->func), c->gridDim.x, c->gridDim.y, c->gridDim.z, c->blockDim.x, c->blockDim.y,
               c->blockDim.z, (unsigned)c->dynamicSmemBytes, rt_pdl(c), (void**)p->args, "rt");
    }
}

// Writes the counts so far and flushes (zrec.py calls it when it dumps; the server is stopped with SIGKILL).
void ztrace_flush(void) {
    pthread_mutex_lock(&mu);
    if (out) {
        fprintf(out, "{\"summary\":{\"launches\":%llu,\"written\":%llu,\"via_rt\":%llu,\"unresolved\":%llu}}\n",
                launches, written, by_rt, unresolved);
        fflush(out);
    }
    pthread_mutex_unlock(&mu);
}

static void finish(void) { ztrace_flush(); }

int InitializeInjection(void) {
    const char* path = getenv("ZTRACE_FILE");
    out = fopen(path ? path : "/tmp/ztrace.jsonl", "w");
    if (!out) return 0;
    setvbuf(out, NULL, _IOFBF, 1 << 20);
    // the process's own runtime (torch loads it globally), never a second copy
    func_by_symbol = (func_by_symbol_t)dlsym(RTLD_DEFAULT, "cudaGetFuncBySymbol");
    if (!func_by_symbol) {
        void* rt = dlopen("libcudart.so.13", RTLD_LAZY | RTLD_NOLOAD);
        if (rt) func_by_symbol = (func_by_symbol_t)dlsym(rt, "cudaGetFuncBySymbol");
    }
    CUpti_SubscriberHandle sub;
    if (cuptiSubscribe(&sub, (CUpti_CallbackFunc)callback, NULL) != CUPTI_SUCCESS) return 0;
    const CUpti_CallbackId drv[] = {CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
                                    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz};
    const CUpti_CallbackId rt[] = {CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060};
    for (size_t i = 0; i < sizeof(drv) / sizeof(drv[0]); ++i) cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_DRIVER_API, drv[i]);
    for (size_t i = 0; i < sizeof(rt) / sizeof(rt[0]); ++i) cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API, rt[i]);
    atexit(finish);
    return 1;
}
