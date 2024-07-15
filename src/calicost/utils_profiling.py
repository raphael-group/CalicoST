try:
    from line_profiler import profile
except:
    def profile(func):
        return func    
