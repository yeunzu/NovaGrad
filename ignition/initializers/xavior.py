from ..backend import Backend

def Xavior(input_size, output_size, backend: Backend):
    # w = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
    w = backend.randn((input_size, output_size)) * backend.sqrt(1 / input_size)
    return w