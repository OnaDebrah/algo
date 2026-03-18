from typing import List

from .....strategies.arbitrage.derivative.garch.base_garch_model import BaseGARCHModel
from .component_garch_model import ComponentGARCHModel
from .egarch_model import EGARCHModel
from .garch_ensemble import GARCHEnsemble
from .garch_model import GARCHModel
from .gjr_garch_model import GJRGARCHModel


class GARCHFactory:
    """Factory for creating GARCH models"""

    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseGARCHModel:
        """Create a GARCH model by type"""

        models = {
            "garch": GARCHModel,
            "egarch": EGARCHModel,
            "gjr": GJRGARCHModel,
            "component": ComponentGARCHModel,
        }

        model_class = models.get(model_type.lower(), GARCHModel)
        return model_class()

    @staticmethod
    def create_ensemble(model_types: List[str] = None) -> GARCHEnsemble:
        """Create an ensemble of GARCH models"""

        if model_types is None:
            model_types = ["garch", "egarch", "gjr", "component"]

        models = [GARCHFactory.create_model(mt) for mt in model_types]
        return GARCHEnsemble(models)
