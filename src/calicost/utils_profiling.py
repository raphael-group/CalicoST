from line_profiler import profile

def profile(func, switch=True):
    if switch:
        return profile(func)
    else:
        return func
    
