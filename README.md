# Gas Concentration Criteria

**Type:** Rule-Based (logic)
**Owner:** dx04
**Auto-generated:** MLOps Platform Model Generator

## Development Workflow

이 모델은 **notebook-first 개발 워크플로우**를 따릅니다:

1. **🧪 Prototype in Jupyter** - `logic.ipynb`에서 실험 및 검증
2. **📝 Productionize** - 검증된 코드를 `logic.py`로 복사
3. **🚀 Deploy** - Airflow를 통해 실행하고 BentoML로 서빙

---

## Quick Start

### Step 1: Open Jupyter Notebook

- Jupyter Lab: http://localhost:8888
- File: `/workspace/models/gas_concentration_criteria/logic.ipynb`

### Step 2: Prototype Your Model

`logic.ipynb`를 열고 가이드된 워크플로우를 따르세요.

### Step 3: Move to Production

검증된 코드를 `logic.py`에 복사하고 `config.yml`의 입력/출력을 적절히 업데이트하세요.

### Step 4: Deploy to Platform

```bash
cd /workspace/infrastructure
docker-compose restart airflow
```

### Step 5: Monitor in Airflow UI

- Airflow UI: http://localhost:9016 (admin/admin)
- Find your DAG: `gas_concentration_criteria`
- Trigger manually to test

## Next Steps

- `logic.py`에서 모델 로직 커스터마이즈
- `config.yml`에서 입력/출력 필드 업데이트
- `requirements.txt`에 의존성 추가
- `config.yml`에서 배포 설정 구성

## Support

- Documentation: See `/workspace/CLAUDE.md`
- Examples: Check `/workspace/models/example_ml_model` or `/workspace/models/example_rule_model`
