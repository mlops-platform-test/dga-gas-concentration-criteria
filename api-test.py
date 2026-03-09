"""
Gas Concentration Criteria BentoML API 테스트 스크립트

Usage:
    python api-test.py
"""

import json
import os
import requests

BENTO_URL = os.getenv("BENTO_URL", "https://mlops.lab.atgdevs.com/gas_concentration_criteria/predict")
MODEL_NAME = "gas_concentration_criteria"


def prepare_request_payload(instances):
    return {"req": {"instances": instances}}


def send_prediction_request(payload):
    try:
        response = requests.post(
            BENTO_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API 요청 실패: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"응답 코드: {e.response.status_code}")
            print(f"응답 내용: {e.response.text}")
        return None


def main():
    print("=" * 70)
    print(f"{MODEL_NAME} BentoML API 테스트 (가스 농도 기준 평가)")
    print("=" * 70)

    test_cases = [
        {"h2": 50.0,   "ch4": 80.0,   "c2h2": 0.3,  "c2h4": 20.0,  "c2h6": 40.0,  "co": 200.0,  "co2": 1500.0},
        {"h2": 200.0,  "ch4": 200.0,  "c2h2": 3.0,  "c2h4": 70.0,  "c2h6": 80.0,  "co": 450.0,  "co2": 3000.0},
        {"h2": 800.0,  "ch4": 500.0,  "c2h2": 15.0, "c2h4": 130.0, "c2h6": 120.0, "co": 600.0,  "co2": 5000.0},
        {"h2": 2000.0, "ch4": 1200.0, "c2h2": 40.0, "c2h4": 250.0, "c2h6": 200.0, "co": 1500.0, "co2": 12000.0},
    ]

    print("\n[1] 테스트 데이터:")
    for i, tc in enumerate(test_cases, 1):
        print(f"   [{i}] H2={tc['h2']}, CH4={tc['ch4']}, C2H2={tc['c2h2']}, "
              f"C2H4={tc['c2h4']}, C2H6={tc['c2h6']}, CO={tc['co']}, CO2={tc['co2']}")

    payload = prepare_request_payload(test_cases)
    print(f"\n[2] BentoML API 호출: {BENTO_URL}")
    response = send_prediction_request(payload)

    if response is None:
        print("\nAPI 요청 실패")
        return

    print("\n[3] 예측 결과:")
    print("-" * 70)

    predictions = response.get('predictions', [])
    if not predictions:
        print("예측 결과가 비어있습니다.")
        print(f"응답 내용: {json.dumps(response, indent=2, ensure_ascii=False)}")
        return

    for i, pred in enumerate(predictions, 1):
        print(f"\n결과 {i}: [{pred.get('max_condition_label', 'N/A')}]")
        print(f"  TCG: {pred.get('tcg', 0):.1f} ppm")
        for gas in ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2", "TCG"]:
            level = pred.get(f"condition_level_{gas}", "N/A")
            print(f"  {gas}: LEVEL_{level}")
        print("  IEC 판정:")
        for gas in ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]:
            print(f"    {gas}: {pred.get(f'iec_{gas}', 'N/A')}")

    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
