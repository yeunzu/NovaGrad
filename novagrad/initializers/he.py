from ..backend import Backend

def He(input_size, output_size, backend: Backend):
    # w = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
    w = backend.randn((input_size, output_size)) * backend.sqrt(2 / input_size)
    return w