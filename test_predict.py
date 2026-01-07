import importlib, traceback

try:
    m = importlib.import_module('main')
    print('Imported main, model:', 'loaded' if getattr(m,'model', None) is not None else 'NOT loaded')
    print('Calling predict_tumor on sample Tr-no_0019.jpg...')
    res = m.predict_tumor(r'..\\SAMPLRMRIIMAGES\\Tr-no_0019.jpg')
    print('predict_tumor returned:', res)
except Exception as e:
    print('ERROR during prediction:')
    traceback.print_exc()
