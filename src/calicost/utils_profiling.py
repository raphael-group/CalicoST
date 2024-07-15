from line_profiler import profile

def profile(func, run_profiler=True):
    if run_profiler:
        def wrapper(func):
            return profile(func)

        return wrapper        
    else:
        return func
    
