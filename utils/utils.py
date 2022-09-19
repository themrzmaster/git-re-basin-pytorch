import copy

def flatten_params(model):
  return model.state_dict()

def lerp(lam, t1, t2):
  t3 = copy.deepcopy(t2)
  for p in t1: 
    t3[p] = (1 - lam) * t1[p] + lam * t2[p]
  return t3