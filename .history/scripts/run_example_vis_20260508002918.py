import runpy, traceback, sys, os

try:
    # Ensure project root is on sys.path so local package imports work
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print('Starting example_visualizations')
    runpy.run_path('examples/example_visualizations.py')
    print('Finished example_visualizations')
except Exception:
    traceback.print_exc()
