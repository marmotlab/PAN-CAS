import torch
import copy

def get_new_pref(old_pref):
    if old_pref[0, 0] != 1:
        old_pref = torch.cat((torch.FloatTensor([1, 0])[None, :].to(old_pref.device), old_pref), 0)
    if old_pref[-1, 0] != 0:
        old_pref = torch.cat((old_pref, torch.FloatTensor([0, 1])[None, :].to(old_pref.device)), 0)

    new_pref = torch.zeros(0, 2).to(old_pref.device)
    for i in range(old_pref.size(0)-1):
        a, b = old_pref[i], old_pref[i+1]
        # insert two new prefs
        c1 = 2/3*a + 1/3*b
        c2 = 1/3*a + 2/3*b
        if i == old_pref.size(0) - 2:
            new_pref = torch.cat((new_pref, a[None, :], c1[None, :], c2[None, :], b[None, :]), 0)
        else:
            new_pref = torch.cat((new_pref, a[None, :], c1[None, :], c2[None, :]), 0)

    return new_pref