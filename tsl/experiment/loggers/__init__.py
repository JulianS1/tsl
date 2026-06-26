try:
    from .neptune_logger import NeptuneLogger
    __all__ = ['NeptuneLogger']
except ImportError:
    NeptuneLogger = None
    __all__ = []

logger_classes = __all__
