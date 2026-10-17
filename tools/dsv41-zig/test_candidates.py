"""Check the built candidate kernel against an independent stable sort; no model load or compilation."""
import ctypes as C
import json
import sys
import time
import numpy as np
import torch


def checked(status):
    if status:
        raise RuntimeError(f'CUDA driver error {status}')


def main():
    torch.cuda.init()
    context_guard = torch.empty(1, device="cuda")
    driver = C.CDLL('libcuda.so.1')
    module, function = C.c_void_p(), C.c_void_p()
    checked(driver.cuModuleLoad(C.byref(module), sys.argv[1].encode()))
    checked(driver.cuModuleGetFunction(C.byref(function), module, b'tf_ds_candidates_kernel'))
    stream = torch.cuda.current_stream()
    rng = np.random.default_rng(20261010)
    results = []

    def select(x, k):
        rows, n = x.shape
        tiles = (n + 4095) // 4096
        partial = torch.empty((rows * tiles * k, 2), dtype=torch.int32, device='cuda')
        output = torch.empty((rows * k, 2), dtype=torch.int32, device='cuda')
        for stage in (0, 1):
            args = [C.c_void_p(x.data_ptr()), C.c_void_p(partial.data_ptr()),
                    C.c_void_p((output if stage else partial).data_ptr()), C.c_uint64(n),
                    C.c_int(n), C.c_int(k), C.c_int(tiles), C.c_int(stage)]
            params = (C.c_void_p * len(args))(*[C.cast(C.byref(a), C.c_void_p) for a in args])
            checked(driver.cuLaunchKernel(function, tiles if stage == 0 else 1, rows, 1,
                                          256, 1, 1, 0, C.c_void_p(stream.cuda_stream), params, None))
        return output.cpu().numpy().reshape(rows, k, 2)

    for rows in (1, 8, 16):
        for n in (17, 4097, 129280):
            for k in (1, 28, 64):
                if k > n:
                    continue
                host = rng.standard_normal((rows, n)).astype(np.float32)
                # Repeated values, ties across tiles, signed zero, extreme finite values.
                host[:, ::7] = 1.0
                host[:, 0] = -0.0
                host[:, 1] = 0.0
                host[:, 2] = np.finfo(np.float32).max
                host[:, 3] = -np.finfo(np.float32).max
                x = torch.from_numpy(host).cuda()
                result = select(x, k)
                for row in range(rows):
                    ids = np.lexsort((np.arange(n), -host[row]))[:k]
                    np.testing.assert_array_equal(result[row, :, 1], ids)
                    np.testing.assert_array_equal(result[row, :, 0].view(np.uint32), host[row, ids].view(np.uint32))
                start = time.perf_counter()
                for _ in range(5):
                    select(x, k)
                results.append({'rows': rows, 'vocab': n, 'k': k, 'selection_copy_ms': (time.perf_counter()-start)*200})
    for value in (float('nan'), float('inf'), -float('inf')):
        host = np.zeros((2, 4097), dtype=np.float32)
        host[1, -1] = value
        got = select(torch.from_numpy(host).cuda(), 28)
        np.testing.assert_array_equal(got[0, :, 1], np.arange(28))
        assert np.all(got[1, :, 1].view(np.uint32) == 0xfffffffe)
    # All finite values equal, with zeros of alternating sign, must pick smallest ids.
    host = np.zeros((1, 129280), dtype=np.float32)
    host[:, ::2] = -0.0
    got = select(torch.from_numpy(host).cuda(), 64)
    np.testing.assert_array_equal(got[0, :, 1], np.arange(64))
    np.testing.assert_array_equal(got[0, :, 0].view(np.uint32), host[0, :64].view(np.uint32))
    checked(driver.cuModuleUnload(module))
    print(json.dumps({'status': 'pass', 'finite_cases':len(results), 'fallback_cases':3,
                      'all_tied_case':1, 'results':results,
                      'timing_limit':'Synthetic selection and readback including Python allocations; live model remained loaded. Not end-to-end generation throughput.'}, indent=2))


if __name__ == '__main__':
    main()
