"""
Gas Concentration Criteria - Rule-Based Logic Implementation

IEEE C57.104-2019 + IEC 60599 기반 가스 농도 기준 평가:
- 4단계 Condition Level 평가 (IEEE C57.104)
- IEC 60599 정상/이상 판정
- TCG (Total Combustible Gas) 계산
"""

from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path

import os
import requests
import pandas as pd
import mlflow
import mlflow.pyfunc

from coreflow.exceptions import MLflowError
from coreflow.utils.logging_helpers import setup_model_logger
from coreflow.utils.mlflow_helpers import init_mlflow, log_deploy_bundle

MODEL_NAME = "gas_concentration_criteria"
logger = setup_model_logger(MODEL_NAME)


def _get_experiment_name() -> str:
    """config.yml의 mlflow.experiment_name을 읽어 반환"""
    import yaml
    config_path = Path(__file__).parent / "config.yml"
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config.get("mlflow", {}).get("experiment_name", MODEL_NAME)
    except Exception:
        return MODEL_NAME


# =============================================================================
# IEEE C57.104 4단계 Condition Level 기준
# =============================================================================

GAS_LIMITS_4CONDITION = {
    "H2":   {"c1max": 100.0,  "c3min": 701.0,  "c3max": 1800.0},
    "CH4":  {"c1max": 120.0,  "c3min": 401.0,  "c3max": 1000.0},
    "C2H2": {"c1max": 1.0,    "c3min": 10.0,   "c3max": 35.0},
    "C2H4": {"c1max": 50.0,   "c3min": 101.0,  "c3max": 200.0},
    "C2H6": {"c1max": 65.0,   "c3min": 101.0,  "c3max": 150.0},
    "CO":   {"c1max": 350.0,  "c3min": 571.0,  "c3max": 1400.0},
    "CO2":  {"c1max": 2500.0, "c3min": 4001.0, "c3max": 10000.0},
    "TCG":  {"c1max": 720.0,  "c3min": 1921.0, "c3max": 4630.0},
}

# IEC 60599 기준
GAS_LIMITS_IEC = {
    "H2":   {"limit": 100,  "mean": "부분방전 징후(100~150ppm)"},
    "CH4":  {"limit": 120,  "mean": "절연지 파열(120ppm)"},
    "C2H2": {"limit": 1,    "mean": "아크방전 또는 중대한 전기적 고장 의심(1ppm)"},
    "C2H4": {"limit": 40,   "mean": "고온 파열(40ppm)"},
    "C2H6": {"limit": 65,   "mean": "중간온도 파열(65ppm)"},
    "CO":   {"limit": 350,  "mean": "절연지(종이) 열화(350~500ppm)"},
    "CO2":  {"limit": 2500, "mean": "절연지(종이) 열화(2500~4000ppm)"},
}


# =============================================================================
# 핵심 평가 로직
# =============================================================================

def _get_condition_level(value: float, limits: Dict[str, float]) -> int:
    """
    IEEE C57.104 4단계 Condition Level 판정.

    LEVEL_1: value < c1max
    LEVEL_2: c1max <= value < c3min
    LEVEL_3: c3min <= value < c3max
    LEVEL_4: value >= c3max
    """
    c1max = limits["c1max"]
    c3min = limits["c3min"]
    c3max = limits["c3max"]

    if value < c1max:
        return 1
    elif value < c3min:
        return 2
    elif value < c3max:
        return 3
    else:
        return 4


def evaluate4ConditionLevel(
    h2: float, ch4: float, c2h2: float, c2h4: float, c2h6: float, co: float, co2: float
) -> Dict[str, Any]:
    """
    IEEE C57.104 기반 4단계 Condition Level 평가.

    Args:
        h2, ch4, c2h2, c2h4, c2h6, co, co2: 가스 농도 (ppm)

    Returns:
        각 가스별 condition level + TCG level + overall max level
    """
    tcg = h2 + ch4 + c2h2 + c2h4 + c2h6 + co

    gas_values = {
        "H2": h2, "CH4": ch4, "C2H2": c2h2,
        "C2H4": c2h4, "C2H6": c2h6, "CO": co, "CO2": co2, "TCG": tcg,
    }

    levels = {}
    for gas, value in gas_values.items():
        levels[gas] = _get_condition_level(value, GAS_LIMITS_4CONDITION[gas])

    max_level = max(levels.values())

    return {
        "levels": levels,
        "tcg": tcg,
        "max_condition_level": max_level,
        "max_condition_label": f"LEVEL_{max_level}",
    }


def evaluateLimitLevel(
    h2: float, ch4: float, c2h2: float, c2h4: float, c2h6: float, co: float, co2: float
) -> Dict[str, str]:
    """
    IEC 60599 기준 정상/이상 판정.

    Args:
        h2, ch4, c2h2, c2h4, c2h6, co, co2: 가스 농도 (ppm)

    Returns:
        각 가스별 판정 결과 (정상 or 이상 설명)
    """
    gas_values = {
        "H2": h2, "CH4": ch4, "C2H2": c2h2,
        "C2H4": c2h4, "C2H6": c2h6, "CO": co, "CO2": co2,
    }

    results = {}
    for gas, value in gas_values.items():
        limit_info = GAS_LIMITS_IEC[gas]
        if value < limit_info["limit"]:
            results[gas] = "정상"
        else:
            results[gas] = f"{limit_info['mean']}, 계측된 농도: {value}ppm"

    return results


def evaluate_gas_concentration(
    h2: float, ch4: float, c2h2: float, c2h4: float, c2h6: float, co: float, co2: float
) -> Dict[str, Any]:
    """
    전체 가스 농도 기준 평가 (4단계 + IEC).

    Returns:
        condition_levels: 각 가스별 condition level
        tcg: Total Combustible Gas (ppm)
        max_condition_level: 최대 condition level (int)
        max_condition_label: 최대 condition level 문자열
        iec_results: IEC 60599 판정 결과
    """
    condition4 = evaluate4ConditionLevel(h2, ch4, c2h2, c2h4, c2h6, co, co2)
    iec = evaluateLimitLevel(h2, ch4, c2h2, c2h4, c2h6, co, co2)

    return {
        "condition_levels": condition4["levels"],
        "tcg": condition4["tcg"],
        "max_condition_level": condition4["max_condition_level"],
        "max_condition_label": condition4["max_condition_label"],
        "iec_results": iec,
    }


# =============================================================================
# MLflow PythonModel 래퍼 (서빙용)
# =============================================================================

class RuleModel(mlflow.pyfunc.PythonModel):
    """가스 농도 기준 평가 로직을 MLflow 모델로 서빙하기 위한 래퍼 클래스"""

    def predict(self, context, model_input):
        if isinstance(model_input, dict):
            model_input = pd.DataFrame([model_input])
        elif not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        return rule_logic(model_input)


def rule_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame 입력을 받아 가스 농도 기준 평가 결과를 DataFrame으로 반환.

    입력 컬럼: h2, ch4, c2h2, c2h4, c2h6, co, co2 (ppm, float)
    출력 컬럼: condition_level_*, tcg, max_condition_level, max_condition_label, iec_*
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    records = []
    for _, row in df.iterrows():
        h2   = float(row.get("h2",   0.0))
        ch4  = float(row.get("ch4",  0.0))
        c2h2 = float(row.get("c2h2", 0.0))
        c2h4 = float(row.get("c2h4", 0.0))
        c2h6 = float(row.get("c2h6", 0.0))
        co   = float(row.get("co",   0.0))
        co2  = float(row.get("co2",  0.0))

        result = evaluate_gas_concentration(h2, ch4, c2h2, c2h4, c2h6, co, co2)

        flat = {
            "tcg": result["tcg"],
            "max_condition_level": result["max_condition_level"],
            "max_condition_label": result["max_condition_label"],
        }
        for gas, level in result["condition_levels"].items():
            flat[f"condition_level_{gas}"] = level
        for gas, status in result["iec_results"].items():
            flat[f"iec_{gas}"] = status

        records.append(flat)

    return pd.DataFrame(records)


# =============================================================================
# Infisical Client
# =============================================================================

class _InfisicalClient:
    """Infisical Universal Auth 기반 secrets 조회 클라이언트."""

    def __init__(self) -> None:
        self.base_url = os.environ["INFISICAL_SITE_URL"].rstrip("/")
        self.client_id = os.environ["INFISICAL_CLIENT_ID"]
        self.client_secret = os.environ["INFISICAL_CLIENT_SECRET"]
        self.project_id = os.environ["INFISICAL_PROJECT_ID"]
        self.environment = os.getenv("INFISICAL_ENV", "dev")
        self._access_token: str | None = None

    def _login(self) -> None:
        resp = requests.post(
            f"{self.base_url}/api/v1/auth/universal-auth/login",
            json={"clientId": self.client_id, "clientSecret": self.client_secret},
            timeout=10,
        )
        resp.raise_for_status()
        self._access_token = resp.json()["accessToken"]

    def _headers(self) -> dict:
        if not self._access_token:
            self._login()
        return {"Authorization": f"Bearer {self._access_token}"}

    def get_secrets(self, path: str) -> dict[str, str]:
        """path 경로의 모든 secrets를 {key: value} dict로 반환."""
        resp = requests.get(
            f"{self.base_url}/api/v3/secrets/raw",
            headers=self._headers(),
            params={
                "workspaceId": self.project_id,
                "environment": self.environment,
                "secretPath": path,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return {s["secretKey"]: s["secretValue"] for s in resp.json().get("secrets", [])}


def _load_secrets_to_env() -> None:
    """Infisical /platform 경로 secrets를 os.environ에 주입.

    Airflow DAG(model_dag_factory.py)에서 INFISICAL_* 환경변수가 K8s Pod에 주입되므로
    execute 스테이지에서 이 함수를 호출하여 /platform 시크릿을 로드한다.

    /platform 경로:
      MLFLOW_TRACKING_URI, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
      AWS_ENDPOINT_URL, MLFLOW_S3_ENDPOINT_URL
    """
    secrets = _InfisicalClient().get_secrets("/platform")
    for key, value in secrets.items():
        os.environ.setdefault(key, value)
    logger.info("Infisical /platform secrets 주입 완료 (count=%d)", len(secrets))


# =============================================================================
# 파이프라인 스테이지
# =============================================================================

def prepare(context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """데이터 준비 — 로직 모델은 외부 데이터 소스 불필요"""
    logger.info("prepare: Gas Concentration Criteria logic model")
    return {}


def execute(data_paths: Dict[str, Any]) -> Dict[str, str]:
    """로직 실행 + MLflow 등록 — K8s Pod에서 실행되는 통합 스테이지

    Args:
        data_paths: prepare() 출력 (로직 모델은 빈 dict)

    Returns:
        {"run_id": "...", "model_uri": "..."}
    """
    try:
        logger.info(f"execute: Gas Concentration Criteria, data_paths={data_paths}")

        # Airflow DAG에서 INFISICAL_* 환경변수가 주입되므로 /platform 시크릿 로드
        _load_secrets_to_env()

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        experiment_name = _get_experiment_name()
        init_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)

        with mlflow.start_run(run_name=f"{MODEL_NAME}_execute") as run:
            from mlflow.models.signature import infer_signature

            example_input = pd.DataFrame([
                {"h2": 80.0,   "ch4": 100.0,  "c2h2": 0.5,  "c2h4": 30.0,  "c2h6": 50.0,  "co": 300.0,  "co2": 2000.0},
                {"h2": 500.0,  "ch4": 300.0,  "c2h2": 5.0,  "c2h4": 80.0,  "c2h6": 90.0,  "co": 500.0,  "co2": 3500.0},
                {"h2": 2000.0, "ch4": 1200.0, "c2h2": 40.0, "c2h4": 250.0, "c2h6": 200.0, "co": 1500.0, "co2": 12000.0},
            ])
            example_output = rule_logic(example_input)
            signature = infer_signature(example_input, example_output)

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=RuleModel(),
                signature=signature,
                registered_model_name=MODEL_NAME,
            )
            mlflow.set_tag("model_kind", "logic")
            mlflow.set_tag("source", "airflow")

            log_deploy_bundle(MODEL_NAME, Path(__file__).parent)

            logger.info(f"execute done. run_id={run.info.run_id}")
            return {
                "run_id": run.info.run_id,
                "model_uri": f"runs:/{run.info.run_id}/model",
            }

    except Exception as e:
        logger.error(f"execute failed: {e}", exc_info=True)
        raise MLflowError(f"execute failed: {e}") from e


if __name__ == "__main__":
    test_cases = [
        {"h2": 50.0,   "ch4": 80.0,   "c2h2": 0.3,  "c2h4": 20.0,  "c2h6": 40.0,  "co": 200.0,  "co2": 1500.0},
        {"h2": 200.0,  "ch4": 200.0,  "c2h2": 3.0,  "c2h4": 70.0,  "c2h6": 80.0,  "co": 450.0,  "co2": 3000.0},
        {"h2": 800.0,  "ch4": 500.0,  "c2h2": 15.0, "c2h4": 130.0, "c2h6": 120.0, "co": 600.0,  "co2": 5000.0},
        {"h2": 2000.0, "ch4": 1200.0, "c2h2": 40.0, "c2h4": 250.0, "c2h6": 200.0, "co": 1500.0, "co2": 12000.0},
    ]

    df_in = pd.DataFrame(test_cases)
    df_out = rule_logic(df_in)

    print("=" * 80)
    print("Gas Concentration Criteria - 테스트 결과")
    print("=" * 80)
    for i, (inp, (_, out)) in enumerate(zip(test_cases, df_out.iterrows())):
        print(f"\n[케이스 {i+1}] H2={inp['h2']}, CH4={inp['ch4']}, C2H2={inp['c2h2']}, "
              f"C2H4={inp['c2h4']}, C2H6={inp['c2h6']}, CO={inp['co']}, CO2={inp['co2']}")
        print(f"  TCG: {out['tcg']:.1f} ppm")
        print(f"  최대 Condition: {out['max_condition_label']}")
        print("  개별 Condition Level:")
        for gas in ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2", "TCG"]:
            print(f"    {gas}: LEVEL_{out[f'condition_level_{gas}']}")
        print("  IEC 판정:")
        for gas in ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]:
            print(f"    {gas}: {out[f'iec_{gas}']}")
