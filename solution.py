import numpy as np
from collections import defaultdict

def num2list(num):
    res = []
    if num == 0:
        return [0]
    while num > 0:
        num, mod = divmod(num, 10)
        res.append(mod)
    return res[::-1]


# считает людей в группах, если n_customers задано
# числом вида x * 10^n, где n любое положительное целое число
# xmax exclusive
def freq(xmax):
    num_list = num2list(xmax)
    lead_dig = num_list[0]
    maxpow = len(num_list) - 1
    smax = 9 * (maxpow + 1)
    if xmax == 0:
        return np.zeros((smax), int)

    if xmax < 10:
        return np.array([1] * xmax + [0] * (smax - xmax))

    fr = [1] * 10 + [0] * (smax + 1 - 10)
    fr = np.array(fr).reshape((1, smax + 1))
    fr = np.concatenate((fr, fr), axis=0)

    for p in range(1, maxpow + 1):
        for i in range(1, 10):
            if p == maxpow and i == lead_dig:
                return fr[-1]
            toadd = np.pad(fr[0, : -i].copy(), (i, 0), mode = 'constant')
            fr[1] += toadd

        fr[0] = fr[-1]
    return np.trim_zeros(fr[-1], 'b')


# считает людей в группах если n_first_id == 0
# xmax exclusive
def allfreq(xmax):
    if xmax == 0:
        return np.zeros((1), int)
    num_list = num2list(xmax)
    lead_dig = num_list[0]
    maxpow = len(num_list) - 1
    smax = 9 * (maxpow + 1)
    offset = 0
    fr = freq(lead_dig * 10 ** maxpow)

    for i in range(maxpow):
        lead_dig = num_list[i + 1]
        offset += num_list[i]
        if lead_dig == 0:
            continue
        num = lead_dig * 10 ** (maxpow - (i + 1))
        tmp = freq(num)
        tmp = np.pad(tmp, (0, smax + 1 - len(tmp)), mode = 'constant')
        tmp = np.roll(tmp, offset)
        fr += tmp

    return np.trim_zeros(fr, 'b')


# Итоговая функция.
# Считает людей в группах с учетом n_first_id
def cust_cnt_in_range(n_customers, n_first_id = 0):
    xmax = n_first_id + n_customers
    fr = allfreq(xmax)
    fr_ = allfreq(n_first_id)
    fr_ = np.pad(fr_, (0, len(fr) - len(fr_)), mode = 'constant')
    res = np.trim_zeros((fr - fr_), 'b')
    ans = {k: v for k, v in zip(range(len(res)), res) if v != 0}
    return ans