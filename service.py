"""
Gas Concentration Criteria - BentoML Service

표준화된 모델 서빙을 위해 BaseModelService를 사용합니다.
"""

import bentoml
from coreflow.serving.base_service import BaseModelService


@bentoml.service(
    name="gas_concentration_criteria_service",
    resources={"cpu": "1"},
)
class ModelService(BaseModelService):
    """Gas Concentration Criteria Service."""

    MODEL_NAME = "gas_concentration_criteria"

    def get_model_name(self) -> str:
        return self.MODEL_NAME

    def get_module_mappings(self):
        """pickle된 logic 모듈을 MLflow unpickling을 위해 매핑"""
        return {
            "models.gas_concentration_criteria.logic": "logic"
        }
