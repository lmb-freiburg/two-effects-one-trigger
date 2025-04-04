import os 

def print_with_rank(*args, **kwargs):
    rank = get_rank()
    world_size = get_world_size()
    print(f"Rank {rank:>2d}/{world_size}:", *args, **kwargs)

def get_rank() -> int:
    """
    In some cases LOCAL_RANK is set, but RANK is unset. Use LOCAL_RANK in that case.
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank


def get_world_size():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return world_size
