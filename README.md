# weather_forecasting

기상청 단기예보(동네예보)와 정면 대결하는 서울·대구 예측 모델.

- 지점: 서울(ASOS 108), 대구(ASOS 143)
- 리드타임: 단기 0~48시간
- 지표: 강수 F1 (1mm 이상을 비로 판정) / 기온·습도·풍속 MAE
- 컴퓨트: Colab L4

## 데이터

| 용도 | 출처 | 스크립트 |
|---|---|---|
| 정답(관측) | 기상청 ASOS 시간자료 (공공데이터포털) | `scripts/download_asos.py` |
| 입력 기상장 | ERA5 (Open-Meteo archive) | `scripts/download_era5.py` |
| 대결 상대 | 동네예보 발표본 | `scripts/collect_kma_forecast.py` (GitHub Actions 3시간 주기 누적) |

`DATA_GO_KR_KEY` 환경변수(공공데이터포털 인증키)가 필요하다. GitHub Actions 쪽은 저장소 secret 으로 등록한다.

```
python scripts/download_era5.py 2018-01-01 2025-12-31
python scripts/download_asos.py 2018-01-01 2025-12-31
```

평가 함수는 `src/metrics.py`. 확률 컷오프는 검증 구간에서 F1 최대로 정하고 테스트 구간에 고정한다.
