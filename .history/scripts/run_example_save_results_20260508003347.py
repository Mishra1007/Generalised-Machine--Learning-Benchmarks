import runpy, traceback, sys, os

try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print('Starting example_save_results')
    runpy.run_path('examples/example_save_results.py', run_name='__main__')
    print('Finished example_save_results')
except Exception:
    traceback.print_exc()
