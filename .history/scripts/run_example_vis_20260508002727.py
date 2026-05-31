import runpy, traceback

try:
    print('Starting example_visualizations')
    runpy.run_path('examples/example_visualizations.py')
    print('Finished example_visualizations')
except Exception:
    traceback.print_exc()
