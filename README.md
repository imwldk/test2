# AI 번역기

글을 입력하고 [영어] [일본어] [베트남어] 버튼을 누르면 해당 언어로 번역해 주는 Streamlit 웹 앱.
번역은 OpenAI `gpt-6.1-sol` 모델(Responses API)을 사용한다. 요구사항은 [PRD.md](PRD.md) 참고.

## 실행

```
pip install -r requirements.txt
cp .env.example .env      # Windows: copy .env.example .env  → OPENAI_API_KEY 등을 채운다
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속한다. 개발·검증은 Python 3.12 / Streamlit 1.62.0에서 했고,
`requirements.txt`의 최소 버전 조합(streamlit 1.46.0 + openai 1.66.0 + python-dotenv 1.0.0)에서도
동작 테스트를 통과했다 (화면 CSS 확인은 1.62.0에서만 함).

## 설정

설정은 **환경 변수(`.env` 포함) → `st.secrets`(배포 환경) → 기본값** 순서로 읽는다. 전체 목록과 기본값은 `.env.example` 참고.

| 이름 | 설명 |
|---|---|
| `OPENAI_API_KEY` | 필수. 공백·줄바꿈·한글이 섞이면 "키 형식" 오류로 안내한다. |
| `OPENAI_MODEL` | 기본 `gpt-6.1-sol`. 코드에 모델명을 박지 않고 여기서만 바꾼다. |
| `APP_PASSWORD` | 선택. 설정하면 비밀번호를 입력해야 사용할 수 있다. **공개 URL이면 설정 권장.** |
| `MAX_INPUT_CHARS`, `OPENAI_MAX_OUTPUT_TOKENS` | 입력 글자 수 / 출력 토큰 상한 (비용 통제) |
| `MIN_INTERVAL_SECONDS`, `MAX_CALLS_PER_SESSION`, `GLOBAL_CALLS_PER_MINUTE` | 호출 간격 / 세션당 / 서버 전체 분당 호출 제한 |

**주의 — 환경 변수가 `.env`보다 우선한다.** 셸이나 배포 환경에 이미 `OPENAI_API_KEY`가 있으면 `.env` 값은 무시된다
(`python-dotenv` 기본 동작). 키를 바꿨는데 반영되지 않으면 환경 변수를 확인한다.

## 배포 (Streamlit Community Cloud 등)

1. `.env`는 올리지 않는다 (`.gitignore`로 제외됨). 올릴 파일: `app.py`, `requirements.txt`, `.env.example`, `.gitignore`, `README.md`.
2. 배포 서비스의 시크릿 설정에 같은 이름으로 등록한다. Streamlit이라면 앱 설정의 Secrets에 TOML로:
   ```
   OPENAI_API_KEY = "sk-..."
   APP_PASSWORD = "정한-비밀번호"
   ```
3. 배포 URL에서 3개 언어 번역이 되는지 확인한다.
4. 각 서비스의 현재 정책(무료 한도, 슬립)은 배포 시점에 공식 문서로 확인한다.

### 공개 배포 시 꼭 알아둘 것 (비용·남용)

- 공개 URL의 앱은 누구나 **내 API 키로** 번역을 호출할 수 있다. 입력·출력 길이와 세션당/서버 전체 호출 수를 제한하지만,
  서버 전체 분당 한도(`GLOBAL_CALLS_PER_MINUTE`)는 모든 방문자 합산이라 **한 명이 소진하면 다른 사람도 1분간 막힌다.**
  방문자를 구분할 수단이 없어서 생기는 한계이므로, 공개 배포에서는 `APP_PASSWORD` 사용을 권장한다.
- OpenAI 대시보드에서 월 사용 한도를 따로 설정한다. 이 앱의 제한은 보조 수단이다.
- 입력한 글은 번역을 위해 OpenAI로 전송된다(요청에는 `store=false`). 민감한 내용은 입력하지 않는다.

## 검증 현황

- 확인됨: 모의 OpenAI 서버와 가짜 키로 한 헤드리스 테스트 및 브라우저(Playwright) 테스트 — 입력 검증, 오류 종류별 안내, 중복 클릭 방지, 키·오류 상세 비노출, XSS, 접속 비밀번호, 모바일·다크 모드.
- **확인되지 않음 (실제 OpenAI 키가 필요)**: 실제 API가 `store`, `reasoning`, `max_output_tokens` 파라미터를 받아들이는지, 번역 품질,
  프롬프트 인젝션 방어. 문제가 생기면 `OPENAI_REASONING_EFFORT=none`, `OPENAI_MAX_OUTPUT_TOKENS=0`으로 해당 파라미터만 끌 수 있다.
