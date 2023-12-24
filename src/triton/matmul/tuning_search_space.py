import triton
import torch
import random


def _get_num_stages_upper_limit():
    cuda_capabilities = torch.cuda.get_device_capability()

    if cuda_capabilities[0] >= 8:
        # A100
        return 5
    else:
        # GPUs (P100, V100, T4) with Cuda compute capabilites below 8.0 only support a max of two stages
        return 2


def get_default_tune_params():
    configs = [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=1,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2,
                      num_warps=2),
    ]
    return configs


def get_advanced_tune_params():
    configs = [
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3,       
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ]  # + get_configs_io_bound()
    return configs


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        triton.Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': block_k,
                                       'GROUP_SIZE_M': 8},
                                      num_stages=num_stages, num_warps=num_warps))
                    # split_k
                    for group_size_m in [2, 4, 8, 16]:
                        configs.append(triton.Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n,
                                                      'BLOCK_SIZE_K': block_k, 'GROUP_SIZE_M': group_size_m},
                                                     num_stages=num_stages, num_warps=num_warps,))
    return configs


def get_search_space(start, end, step):
    space = []
    i = start
    while (i <= end):
        space.append(i)
        i += step
    return space


def get_random_tune_params():
    configs = []
    i = 0

    block_m_n_space = get_search_space(8, 128, 8)
    block_k_space = get_search_space(8, 64, 4)
    group_m_space = get_search_space(2, 8, 1)
    num_stages_space = get_search_space(1, 2, 1)  # change this to detect max
    print(num_stages_space)
    num_warps_space = get_search_space(2, 8, 2)

    while (i < 200):
        i += 1
        block_m = block_m_n_space[random.randrange(0, len(block_m_n_space) - 1)]
        block_n = block_m_n_space[random.randrange(0, len(block_m_n_space) - 1)]
        block_k = block_k_space[random.randrange(0, len(block_k_space) - 1)]
        group_m = group_m_space[random.randrange(0, len(group_m_space) - 1)]
        num_stages = num_stages_space[random.randrange(0, len(num_stages_space) - 1)]
        num_warps = num_warps_space[random.randrange(0, len(num_warps_space) - 1)]

        configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                                      'GROUP_SIZE_M': group_m},
                                     num_stages=num_stages, num_warps=num_warps))

    return configs
