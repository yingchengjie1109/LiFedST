import random

nums = list(range(0, 227))
random.shuffle(nums)
m = 16  # num_client

n = len(nums) // m
lists = [nums[i * n: (i + 1) * n] for i in range(m)]
if len(nums) % m != 0:
    lists[0].extend(nums[m * n:])

lists = [sorted(lst) for lst in lists]

print(lists)