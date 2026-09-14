from fastapi import FastAPI, Request, BackgroundTasks, Header, HTTPException
from fastapi.responses import HTMLResponse
from crewai import Agent, Task, Crew, Process
import requests
import os
import sys
import asyncio
import json

# ==========================================
# 🚨 인코딩 설정 (한글 깨짐 방지)
# ==========================================
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# 1. API 키 셋업 (본인 키로 변경 필수!)
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY")
model = "gemini/gemini-flash-lite-latest"

app = FastAPI()

# 🧠 사용자 상태(세션)를 기억하는 메모리 저장소
user_sessions = {}

# ==========================================
# 🌟 홍소장 채널 공통 지시사항
# ==========================================
common_guidelines = """
[필수 고려 및 참고 사항]
1. 투명성과 신뢰: 우리는 절대 허위 매물이나 단점을 숨기지 않습니다. 사고 유무와 성능기록부를 투명하게 고지하는 것을 항상 강조하세요.
2. 타깃 시청자: 자동차 전문가가 아닌 일반인입니다. 어려운 전문 용어는 초보자도 이해하기 쉽게 풀어서 설명하세요.
3. 톤앤매너: 적당히 진지하지만 너무 딱딱하지는 않게, 확신에 찬 목소리로 작성하세요.
4. 금지어: '아마도', '비교적', '거의' 같은 애매한 표현은 절대 사용하지 마세요.
5. 영상의 길이는 10분 내외
"""

# 1번 요원: 차량 제원 분석가
analyst = Agent(
    role='중고차 제원 분석가',
    goal='차량의 스펙과 옵션을 분석하여 타깃 고객층과 소싱 포인트를 도출합니다.',
    backstory=f'데이터 분석 전문가로서, 차량의 감가방어율과 실질적인 장점을 객관적으로 파악합니다.\n\n{common_guidelines}',
    llm=model,
    verbose=True
)

# 2번 요원: 대본 구조 기획자
planner = Agent(
    role='유튜브 대본 구조 기획자',
    goal='10분 분량의 롱폼 영상에 맞는 오프닝-인트로(여기에서 차량 주요 정보 및 옵션을 화면에 띄운 상태로 이야기하는 부분)-외관-실내-엔진룸-클로징의 시간 배분과 구조를 짭니다.',
    backstory=f'수백만 조회수를 만드는 유튜브 기획자입니다. 시청자가 이탈하지 않는 완벽한 텐션을 설계합니다. 차량가격은 클로징 단계에서 공개합니다. 오프닝에서 매물에 구매자들이 매력적으로 느낄 만한 Salepoint를 몇 가지 넣어줘 \n\n{common_guidelines}',
    llm=model,
    verbose=True
)

# 3번 요원: 홍소장 페르소나 작가
writer = Agent(
    role='홍소장 페르소나 스토리텔러',
    goal='기획된 구조와 분석 데이터를 바탕으로 홍소장 특유의 신뢰감 있고 시원시원한 대본을 작성합니다.',
    backstory=f'경원모터스 홍성우 소장님의 빙의자입니다. "제가 항상 투명하게 말씀드리죠" 같은 특유의 화법을 완벽히 구사합니다. 나이가 60대이라서 진중한 말투를 사용합니다.\n\n{common_guidelines}',
    llm=model,
    verbose=True
)

task_analyze = Task(
    description="다음 차량 데이터를 분석하여 타깃 고객층과 주요 세일즈 포인트를 3가지로 정리하세요: {car_data}",
    expected_output="타깃 고객층 분석 및 핵심 세일즈 포인트 3가지",
    agent=analyst
)

task_plan = Task(
    description="분석 결과를 바탕으로 10분짜리 유튜브 롱폼 영상의 대본 뼈대(오프닝, 외관, 실내, 가격공개/클로징)를 기획하세요.",
    expected_output="시간 배분이 포함된 상세한 대본 목차 및 구조",
    agent=planner
)

task_write = Task(
    description="기획된 뼈대에 살을 붙여, 중고차 딜러 '홍소장'의 말투로 꽉 찬 대본을 작성하세요. 옵션의 실사용 가치를 강조하세요.",
    expected_output="홍소장 페르소나가 적용된 초안 대본",
    agent=writer
)

task_rewrite = Task(
    description="""다음 [이전 대본]을 읽고, [사용자 피드백]을 적극적으로 반영하여 대본을 처음부터 끝까지 완성된 형태로 다시 작성해주세요.     

    [이전 대본]
    {last_script}
    
    [사용자 피드백]
    {feedback}
    """,
    expected_output="피드백이 반영된 새로운 최종 대본",
    agent=writer
)

# ==========================================
# 🚀 1. 최초 대본 생성 Crew
# ==========================================
async def generate_youtube_script(car_data: str) -> str:
    tasks = [
        task_analyze, task_plan, task_write
    ]
    
    youtube_crew = Crew(agents=[analyst, planner, writer], tasks=tasks, process=Process.sequential, verbose=False)
    result = await youtube_crew.kickoff_async(inputs={'car_data': car_data})
    return result.raw

# ==========================================
# 🚀 2. 피드백 반영(수정) Crew
# ==========================================
async def rewrite_youtube_script(last_script: str, feedback: str) -> str:
    feedback_crew = Crew(agents=[writer], tasks=[task_rewrite], process=Process.sequential, verbose=False)
    result = await feedback_crew.kickoff_async(inputs={'last_script': last_script, 'feedback': feedback})
    return result.raw

# ==========================================
# 🛠️ 카카오 전송 공통 함수 (분할 없이 한 번에 전송)
# ==========================================
def send_to_kakao(callback_url: str, text: str):
    # 완성된 대본과 피드백 안내 문구를 하나의 텍스트로 합칩니다.
    final_text = text + "\n\n✏️ 수정할 부분이 있다면 채팅으로 바로 쳐주세요!\n(예: 오프닝 멘트를 더 신나게 바꿔줘)"
    
    # 하나의 말풍선(simpleText)에 모두 담습니다.
    outputs = [{"simpleText": {"text": final_text}}]

    # 하단 퀵 리플라이(바로가기) 버튼 추가
    quick_replies = [
        {"action": "message", "label": "🔄 처음부터 다시 생성", "messageText": "대본 생성"}
    ]

    callback_payload = {
        "version": "2.0",
        "template": {
            "outputs": outputs,
            "quickReplies": quick_replies
        }
    }
    
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    payload_json = json.dumps(callback_payload, ensure_ascii=False).encode('utf-8')
    requests.post(callback_url, data=payload_json, headers=headers)

# ==========================================
# 🛠️ 백그라운드 작업 래퍼
# ==========================================
def run_crew_background(user_id: str, utterance: str, callback_url: str, is_feedback: bool):
    async def _async_task():
        try:
            if is_feedback:
                print(f"[{user_id}] 피드백 반영 중: {utterance}")
                last_script = user_sessions[user_id]["last_script"]
                final_script = await rewrite_youtube_script(last_script, utterance)
            else:
                print(f"[{user_id}] 신규 대본 생성 중...")
                final_script = await generate_youtube_script(utterance)
            
            # 생성된 대본을 메모리에 저장하여 다음 피드백에 대비함
            user_sessions[user_id] = {"last_script": final_script}
            send_to_kakao(callback_url, final_script)
            
        except Exception as e:
            send_to_kakao(callback_url, f"❌ 에러가 발생했습니다:\n{str(e)}")

    asyncio.run(_async_task())
    
# ==========================================
# 🚀 API 엔드포인트 (보안 기능 추가)
# ==========================================
@app.post("/api/chat")
async def kakao_chat(
    request: Request, 
    background_tasks: BackgroundTasks,
    # 👇 헤더(x-kakao-bot-token) 검사 변수 추가
    x_kakao_bot_token: str = Header(None) 
):
    # 🚨 1. 문지기 보안 검사 (무단 접근 차단)
    # 서버에 저장된 내 비밀번호 가져오기
    saved_token = os.environ.get("KAKAO_BOT_TOKEN")
    
    if saved_token: # 서버에 비밀번호가 세팅되어 있을 때만 검사
        if x_kakao_bot_token != saved_token:
            # 비밀번호가 틀리거나 아예 안 보냈다면 401 Unauthorized 에러를 던지며 접근 거부!
            print(f"🚨 [보안 경고] 비정상적인 접근 시도 차단됨 (입력된 토큰: {x_kakao_bot_token})")
            raise HTTPException(status_code=401, detail="Unauthorized Request")

    # 👇 검문소를 통과한 정상적인 카카오톡 요청만 아래 로직 실행
    payload = await request.json()
    user_id = payload.get("userRequest", {}).get("user", {}).get("id", "unknown")
    utterance = payload.get("userRequest", {}).get("utterance", "").strip()
    callback_url = payload.get("userRequest", {}).get("callbackUrl")

    # 2. 초기화 / 시작 조건
    if utterance == "대본 생성" or utterance == "처음부터 다시 생성":
        user_sessions.pop(user_id, None)
        return {
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": "🚗 차량 정보를 입력해주세요.\n(예: 21년 4월식 K7 프리미어 가솔린 무사고...)"}}]
            }
        }

    if not utterance:
        return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "텍스트를 입력해주세요."}}]}}

    if not callback_url:
        return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "오픈빌더 설정에서 콜백이 꺼져있습니다."}}]}}

    # 3. 사용자 세션 파악
    is_feedback = user_id in user_sessions and "last_script" in user_sessions[user_id]

    if is_feedback:
        background_tasks.add_task(run_crew_background, user_id, utterance, callback_url, is_feedback=True)
        response_text = "🛠️ 요청하신 피드백을 반영하여 대본을 수정하고 있습니다. 잠시만 기다려주세요!"
    else:
        background_tasks.add_task(run_crew_background, user_id, utterance, callback_url, is_feedback=False)
        response_text = "📝 홍소장님 대본 생성을 시작하였습니다.\n약 1~2분 정도 소요됩니다. 잠시만 기다려주세요!"

    return {
        "version": "2.0",
        "useCallback": True, 
        "template": {
            "outputs": [{"simpleText": {"text": response_text}}]
        }
    }

@app.get("/wakeup", response_class=HTMLResponse)
async def wakeup_server():
    """서버 깨우기 전용 웹페이지"""
    return """
    <html>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <body style="display:flex; justify-content:center; align-items:center; height:100vh; background-color:#fef01b; margin:0; font-family:sans-serif; text-align:center;">
            <div style="background:white; padding:30px; border-radius:15px; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="color:#333;">🟢 챗봇 서버 기동 완료!</h2>
                <p style="color:#666; line-height:1.6;">서버가 성공적으로 깨어났습니다.<br>이제 우측 상단의 <b>[ X ]</b>를 눌러 창을 닫고,<br>카톡방에서 <b>'대본 생성'</b>을 눌러주세요!</p>
            </div>
        </body>
    </html>
    """
