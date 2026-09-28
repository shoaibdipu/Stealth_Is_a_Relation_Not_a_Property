from common.runner import aggregate
for d in ['nmnist','dvsgesture','cifar10dvs','ncaltech101','dailydvs200']:
    try: aggregate(d)
    except Exception as e: print(d,'FAILED',repr(e))
