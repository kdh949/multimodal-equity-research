# Git Workflow

이 프로젝트는 GitHub Flow를 따른다.

## Branches

- `main`: 항상 실행 가능한 기준선
- `feature/bootstrap-quant-research-app`: 초기 앱 구현
- 이후 기능은 `feature/<short-name>`, 수정은 `fix/<short-name>` 사용

## Commit Convention

커밋 메시지는 아래 형식을 사용한다.

```text
<type>: <한국어 제목>
```

스코프는 쓰지 않는다.

허용 타입:

- `feat`: 사용자 관점의 기능 추가
- `fix`: 버그 수정
- `refactor`: 동작 변화 없는 구조 개선
- `docs`: 문서 변경
- `test`: 테스트 추가 또는 수정
- `chore`: 유지보수
- `style`: 동작에 영향 없는 포맷팅
- `perf`: 성능 개선
- `build`: 빌드/패키지 설정
- `ci`: CI 설정
- `revert`: 이전 커밋 되돌리기

## Initial Commits

```text
docs: 퀀트 리서치 앱 개발 기준 문서 추가
build: 파이썬 프로젝트 초기 설정 추가
feat: 멀티모달 퀀트 리서치 파이프라인 추가
test: 워크포워드 검증 테스트 추가
```

## Pull Request Template

```markdown
## 무엇을 변경했나요?

## 왜 변경했나요?

## 어떻게 구현했나요?

## 테스트

- [ ] 단위 테스트
- [ ] 통합 테스트
- [ ] Streamlit 수동 확인

## 체크리스트

- [ ] LLM이 매매 결정을 내리지 않음
- [ ] 미래 데이터 누수 방지
- [ ] SEC User-Agent와 rate limit 적용
- [ ] 캐시/아티팩트 미커밋
```
