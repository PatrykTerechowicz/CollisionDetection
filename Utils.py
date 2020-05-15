import torch
def dyskretyzacja(v):
    left = v[0]
    center = v[1]
    right = v[2]
    no_colliion = v[3]
    if no_colliion > left + center + right:
        return [0, 0, 0, 1]
    if left > 3*(right + center)/2:
        return [1, 0, 0, 0]
    if right > 3*(left + center)/2:
        return [0, 0, 1, 0]
    if center > 3*(left + right)/2:
        return [0, 1, 0, 0]
    if center + right > 3*left:
        return [0, 1, 1, 0]
    if left + center > 3*right:
        return [1, 1, 0, 0]
    if left + right > 3*center:
        return [1, 0, 1, 0]
    return [1, 1, 1, 0]


def xor(a, b):
    if a != b:
        return 1.0
    return 0.0

def hamming(output, desired):
    assert len(output) == len(desired)
    sum = 0
    for i in range(len(output)):
        sum = sum + xor(output[i].item(), desired[i])
    return sum
