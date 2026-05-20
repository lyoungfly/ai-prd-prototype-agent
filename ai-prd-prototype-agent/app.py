import os
import json
import sqlite3
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI


# =========================
# 初始化
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if not OPENAI_API_KEY:
    raise RuntimeError("请在 .env 文件中配置 OPENAI_API_KEY")

if OPENAI_BASE_URL:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="AI 产品需求到原型自动生成 Agent")

DB_PATH = "projects.db"


# =========================
# 数据库
# =========================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        title TEXT,
        raw_idea TEXT,
        clarification TEXT,
        prd TEXT,
        user_stories TEXT,
        pages TEXT,
        prototype_html TEXT,
        review TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


init_db()


def save_project(data: Dict[str, Any]):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO projects (
        id,
        title,
        raw_idea,
        clarification,
        prd,
        user_stories,
        pages,
        prototype_html,
        review,
        created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["id"],
        data["title"],
        data["raw_idea"],
        data.get("clarification", ""),
        data.get("prd", ""),
        data.get("user_stories", ""),
        data.get("pages", ""),
        data.get("prototype_html", ""),
        data.get("review", ""),
        data["created_at"]
    ))

    conn.commit()
    conn.close()


def get_project(project_id: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT 
        id,
        title,
        raw_idea,
        clarification,
        prd,
        user_stories,
        pages,
        prototype_html,
        review,
        created_at
    FROM projects WHERE id = ?
    """, (project_id,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "title": row[1],
        "raw_idea": row[2],
        "clarification": row[3],
        "prd": row[4],
        "user_stories": row[5],
        "pages": row[6],
        "prototype_html": row[7],
        "review": row[8],
        "created_at": row[9]
    }


def list_projects() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, title, raw_idea, created_at
    FROM projects
    ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "title": r[1],
            "raw_idea": r[2],
            "created_at": r[3]
        }
        for r in rows
    ]


# =========================
# 请求模型
# =========================

class GenerateRequest(BaseModel):
    idea: str
    target_users: Optional[str] = ""
    business_goal: Optional[str] = ""
    platform: Optional[str] = "Web"
    constraints: Optional[str] = ""


class IterateRequest(BaseModel):
    project_id: str
    feedback: str


# =========================
# AI 通用调用
# =========================

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {str(e)}")


# =========================
# Agent 1：需求澄清 Agent
# =========================

def clarification_agent(
    idea: str,
    target_users: str,
    business_goal: str,
    platform: str,
    constraints: str
) -> str:
    system_prompt = """
你是一个资深产品经理，擅长将模糊想法变成清晰产品需求。
你的任务是分析用户输入的产品创意，并完成需求澄清。

要求：
1. 不要只提出问题，要主动补全合理假设。
2. 如果信息不足，最多提出 5 个关键澄清问题。
3. 输出结构化 Markdown。
4. 内容包括：
   - 产品一句话描述
   - 目标用户假设
   - 核心业务目标
   - 主要使用场景
   - 关键约束
   - 待确认问题
"""

    user_prompt = f"""
产品想法：
{idea}

目标用户：
{target_users}

业务目标：
{business_goal}

平台：
{platform}

约束条件：
{constraints}
"""

    return call_llm(system_prompt, user_prompt)


# =========================
# Agent 2：PRD Agent
# =========================

def prd_agent(clarification: str) -> str:
    system_prompt = """
你是一个高级产品经理，请基于需求澄清内容生成一份专业 PRD。

输出格式必须是 Markdown。

PRD 需要包含：
1. 产品背景
2. 产品目标
3. 目标用户
4. 用户痛点
5. 产品范围
6. 核心功能列表
7. 功能优先级，使用 P0/P1/P2
8. 关键业务流程
9. 数据对象设计
10. 权限与角色
11. 非功能需求
12. 埋点指标
13. 验收标准
14. 风险与边界
"""

    user_prompt = f"""
请根据以下需求澄清内容生成 PRD：

{clarification}
"""

    return call_llm(system_prompt, user_prompt)


# =========================
# Agent 3：用户故事 Agent
# =========================

def user_story_agent(prd: str) -> str:
    system_prompt = """
你是敏捷研发团队中的产品负责人。
请根据 PRD 生成用户故事和验收标准。

输出 Markdown。

每条用户故事使用格式：
- 作为 <角色>
- 我希望 <能力>
- 以便 <价值>
- 验收标准：
  - Given ...
  - When ...
  - Then ...

请按照功能模块分组。
"""

    user_prompt = f"""
PRD 内容：

{prd}
"""

    return call_llm(system_prompt, user_prompt)


# =========================
# Agent 4：页面与信息架构 Agent
# =========================

def page_architecture_agent(prd: str, user_stories: str) -> str:
    system_prompt = """
你是资深 UX 设计师和信息架构师。
请根据 PRD 和用户故事设计产品页面结构。

输出 Markdown。

必须包含：
1. 页面清单
2. 每个页面的目标
3. 页面主要模块
4. 页面字段
5. 关键按钮
6. 交互说明
7. 页面跳转关系
8. 空状态、错误状态、加载状态
"""

    user_prompt = f"""
PRD：
{prd}

用户故事：
{user_stories}
"""

    return call_llm(system_prompt, user_prompt)


# =========================
# Agent 5：HTML 原型 Agent
# =========================

def prototype_agent(prd: str, pages: str) -> str:
    system_prompt = """
你是一个资深前端工程师和产品设计师。
请根据 PRD 和页面结构生成一个可运行的单文件 HTML 高保真原型。

强制要求：
1. 只输出完整 HTML 代码，不要输出解释。
2. 必须包含 <!DOCTYPE html>。
3. 使用内联 CSS 和原生 JavaScript。
4. 不要依赖外部 CDN。
5. 视觉风格现代、简洁、类似 SaaS 产品。
6. 页面宽度适配桌面端。
7. 需要有左侧导航、顶部栏、主内容区。
8. 需要包含多个页面或模块，可通过 JS 切换。
9. 按钮、表单、卡片、表格需要有交互反馈。
10. 用模拟数据展示真实效果。
11. 不允许使用 React/Vue，只能原生 HTML/CSS/JS。
"""

    user_prompt = f"""
PRD：
{prd}

页面结构：
{pages}

请生成完整单文件 HTML 原型。
"""

    html = call_llm(system_prompt, user_prompt, temperature=0.3)

    # 清理模型可能输出的 ```html
    html = html.replace("```html", "").replace("```", "").strip()

    return html


# =========================
# Agent 6：评审 Agent
# =========================

def review_agent(prd: str, pages: str, prototype_html: str) -> str:
    system_prompt = """
你是由产品负责人、技术负责人、UX 负责人和商业分析师组成的评审委员会。
请对当前方案进行综合评审。

输出 Markdown。

必须包含：
1. 总体评分，满分 10 分
2. 商业价值评分
3. 用户体验评分
4. 技术可行性评分
5. MVP 可落地性评分
6. 主要优点
7. 主要问题
8. 高风险点
9. 优先修改建议
10. 下一步研发任务拆解
"""

    user_prompt = f"""
PRD：
{prd}

页面结构：
{pages}

HTML 原型片段：
{prototype_html[:6000]}
"""

    return call_llm(system_prompt, user_prompt)


# =========================
# 标题生成 Agent
# =========================

def title_agent(idea: str) -> str:
    system_prompt = """
你是产品命名助手。
请根据用户的产品想法，生成一个简短中文项目标题。
要求：
1. 不超过 15 个字。
2. 不要解释。
3. 不要加引号。
"""

    return call_llm(system_prompt, idea, temperature=0.5)


# =========================
# 迭代 Agent
# =========================

def iteration_agent(project: Dict[str, Any], feedback: str) -> Dict[str, str]:
    system_prompt = """
你是一个产品迭代 Agent。
用户会给出对已有 PRD 和原型的修改意见。
你需要重新生成：
1. PRD
2. 用户故事
3. 页面结构
4. HTML 原型
5. 评审意见

注意：
- 保留原有合理设计。
- 根据用户反馈做针对性修改。
- 输出必须是 JSON。
- JSON 字段包括：
  - prd
  - user_stories
  - pages
  - prototype_prompt
"""

    user_prompt = f"""
原始产品想法：
{project["raw_idea"]}

原 PRD：
{project["prd"]}

原用户故事：
{project["user_stories"]}

原页面结构：
{project["pages"]}

用户修改意见：
{feedback}

请返回 JSON。
"""

    raw = call_llm(system_prompt, user_prompt, temperature=0.3)

    try:
        raw_clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw_clean)
    except Exception:
        raise HTTPException(status_code=500, detail=f"迭代 Agent 返回 JSON 解析失败：{raw}")

    new_prd = data.get("prd", "")
    new_user_stories = data.get("user_stories", "")
    new_pages = data.get("pages", "")

    new_html = prototype_agent(new_prd, new_pages)
    new_review = review_agent(new_prd, new_pages, new_html)

    return {
        "prd": new_prd,
        "user_stories": new_user_stories,
        "pages": new_pages,
        "prototype_html": new_html,
        "review": new_review
    }


# =========================
# API
# =========================

@app.post("/api/generate")
def generate(req: GenerateRequest):
    if not req.idea.strip():
        raise HTTPException(status_code=400, detail="idea 不能为空")

    title = title_agent(req.idea)

    clarification = clarification_agent(
        idea=req.idea,
        target_users=req.target_users or "",
        business_goal=req.business_goal or "",
        platform=req.platform or "Web",
        constraints=req.constraints or ""
    )

    prd = prd_agent(clarification)
    user_stories = user_story_agent(prd)
    pages = page_architecture_agent(prd, user_stories)
    prototype_html = prototype_agent(prd, pages)
    review = review_agent(prd, pages, prototype_html)

    project_id = str(uuid.uuid4())

    data = {
        "id": project_id,
        "title": title,
        "raw_idea": req.idea,
        "clarification": clarification,
        "prd": prd,
        "user_stories": user_stories,
        "pages": pages,
        "prototype_html": prototype_html,
        "review": review,
        "created_at": datetime.utcnow().isoformat()
    }

    save_project(data)

    return data


@app.get("/api/projects")
def api_list_projects():
    return {
        "items": list_projects()
    }


@app.get("/api/projects/{project_id}")
def api_get_project(project_id: str):
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project


@app.post("/api/iterate")
def iterate(req: IterateRequest):
    project = get_project(req.project_id)

    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    if not req.feedback.strip():
        raise HTTPException(status_code=400, detail="feedback 不能为空")

    result = iteration_agent(project, req.feedback)

    new_project_id = str(uuid.uuid4())

    data = {
        "id": new_project_id,
        "title": project["title"] + " 迭代版",
        "raw_idea": project["raw_idea"] + "\n\n迭代意见：" + req.feedback,
        "clarification": project["clarification"],
        "prd": result["prd"],
        "user_stories": result["user_stories"],
        "pages": result["pages"],
        "prototype_html": result["prototype_html"],
        "review": result["review"],
        "created_at": datetime.utcnow().isoformat()
    }

    save_project(data)

    return data


@app.get("/prototype/{project_id}", response_class=HTMLResponse)
def view_prototype(project_id: str):
    project = get_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    return HTMLResponse(project["prototype_html"])


# =========================
# Web 前端
# =========================

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>AI 产品需求到原型自动生成 Agent</title>
    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif;
            background: #f5f7fb;
            color: #1f2937;
        }

        header {
            height: 64px;
            background: #111827;
            color: white;
            display: flex;
            align-items: center;
            padding: 0 32px;
            font-size: 20px;
            font-weight: 700;
        }

        .container {
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 20px;
            padding: 20px;
        }

        .panel {
            background: white;
            border-radius: 14px;
            padding: 20px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        }

        label {
            display: block;
            margin-top: 14px;
            margin-bottom: 6px;
            font-weight: 600;
            font-size: 14px;
        }

        textarea, input, select {
            width: 100%;
            border: 1px solid #d1d5db;
            border-radius: 10px;
            padding: 10px 12px;
            font-size: 14px;
            outline: none;
        }

        textarea {
            min-height: 120px;
            resize: vertical;
        }

        textarea:focus, input:focus, select:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
        }

        button {
            margin-top: 16px;
            width: 100%;
            border: none;
            border-radius: 10px;
            padding: 12px 16px;
            background: #2563eb;
            color: white;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        button.secondary {
            background: #374151;
        }

        button.secondary:hover {
            background: #1f2937;
        }

        .tabs {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }

        .tab {
            padding: 8px 12px;
            background: #eef2ff;
            color: #3730a3;
            border-radius: 999px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
        }

        .tab.active {
            background: #2563eb;
            color: white;
        }

        pre {
            background: #0f172a;
            color: #e5e7eb;
            padding: 18px;
            border-radius: 12px;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 620px;
            overflow: auto;
            font-size: 13px;
            line-height: 1.6;
        }

        iframe {
            width: 100%;
            height: 720px;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            background: white;
        }

        .loading {
            display: none;
            margin-top: 12px;
            color: #2563eb;
            font-weight: 600;
        }

        .project-list {
            margin-top: 20px;
        }

        .project-item {
            padding: 10px;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            margin-top: 8px;
            cursor: pointer;
            background: #f9fafb;
        }

        .project-item:hover {
            background: #eef2ff;
        }

        .small {
            color: #6b7280;
            font-size: 12px;
            margin-top: 4px;
        }

        .iterate-box {
            margin-top: 18px;
            border-top: 1px solid #e5e7eb;
            padding-top: 16px;
        }
    </style>
</head>
<body>
<header>
    AI 产品需求到原型自动生成 Agent
</header>

<div class="container">
    <div class="panel">
        <h2>输入产品想法</h2>

        <label>产品想法</label>
        <textarea id="idea" placeholder="例如：我想做一个帮助销售团队自动整理客户跟进记录，并生成下一步行动建议的 AI 工具。"></textarea>

        <label>目标用户</label>
        <input id="targetUsers" placeholder="例如：B2B 销售、销售主管">

        <label>业务目标</label>
        <input id="businessGoal" placeholder="例如：提升销售跟进效率，减少 CRM 填写时间">

        <label>平台</label>
        <select id="platform">
            <option value="Web">Web</option>
            <option value="Mobile App">Mobile App</option>
            <option value="小程序">小程序</option>
            <option value="桌面应用">桌面应用</option>
        </select>

        <label>约束条件</label>
        <textarea id="constraints" placeholder="例如：MVP 需要 2 周内完成，只接入企业微信和飞书。"></textarea>

        <button onclick="generate()">生成 PRD 和原型</button>
        <div id="loading" class="loading">正在生成中，可能需要 1-3 分钟，请稍候...</div>

        <div class="iterate-box">
            <h3>基于当前项目迭代</h3>
            <textarea id="feedback" placeholder="例如：增加团队看板页面，支持主管查看每个销售的客户风险。"></textarea>
            <button class="secondary" onclick="iterate()">生成迭代版</button>
        </div>

        <div class="project-list">
            <h3>历史项目</h3>
            <div id="projectList"></div>
        </div>
    </div>

    <div class="panel">
        <div class="tabs">
            <div class="tab active" onclick="showTab('clarification')">需求澄清</div>
            <div class="tab" onclick="showTab('prd')">PRD</div>
            <div class="tab" onclick="showTab('user_stories')">用户故事</div>
            <div class="tab" onclick="showTab('pages')">页面结构</div>
            <div class="tab" onclick="showTab('review')">评审</div>
            <div class="tab" onclick="showTab('prototype')">原型预览</div>
            <div class="tab" onclick="showTab('html')">HTML代码</div>
        </div>

        <div id="content">
            <pre id="textOutput">请先输入产品想法并点击生成。</pre>
        </div>
    </div>
</div>

<script>
    let currentProject = null;
    let currentTab = "clarification";

    async function generate() {
        const idea = document.getElementById("idea").value.trim();
        const targetUsers = document.getElementById("targetUsers").value.trim();
        const businessGoal = document.getElementById("businessGoal").value.trim();
        const platform = document.getElementById("platform").value;
        const constraints = document.getElementById("constraints").value.trim();

        if (!idea) {
            alert("请输入产品想法");
            return;
        }

        document.getElementById("loading").style.display = "block";

        try {
            const res = await fetch("/api/generate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    idea,
                    target_users: targetUsers,
                    business_goal: businessGoal,
                    platform,
                    constraints
                })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "生成失败");
            }

            currentProject = await res.json();
            showTab("clarification");
            loadProjects();

        } catch (e) {
            alert(e.message);
        } finally {
            document.getElementById("loading").style.display = "none";
        }
    }

    async function iterate() {
        if (!currentProject) {
            alert("请先选择或生成一个项目");
            return;
        }

        const feedback = document.getElementById("feedback").value.trim();

        if (!feedback) {
            alert("请输入迭代意见");
            return;
        }

        document.getElementById("loading").style.display = "block";

        try {
            const res = await fetch("/api/iterate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    project_id: currentProject.id,
                    feedback
                })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "迭代失败");
            }

            currentProject = await res.json();
            showTab("prd");
            loadProjects();

        } catch (e) {
            alert(e.message);
        } finally {
            document.getElementById("loading").style.display = "none";
        }
    }

    function showTab(tab) {
        currentTab = tab;

        document.querySelectorAll(".tab").forEach(el => {
            el.classList.remove("active");
        });

        const tabs = document.querySelectorAll(".tab");
        const names = ["clarification", "prd", "user_stories", "pages", "review", "prototype", "html"];
        const index = names.indexOf(tab);
        if (index >= 0) {
            tabs[index].classList.add("active");
        }

        const content = document.getElementById("content");

        if (!currentProject) {
            content.innerHTML = '<pre>请先输入产品想法并点击生成。</pre>';
            return;
        }

        if (tab === "prototype") {
            content.innerHTML = `<iframe src="/prototype/${currentProject.id}"></iframe>`;
            return;
        }

        if (tab === "html") {
            content.innerHTML = `<pre>${escapeHtml(currentProject.prototype_html || "")}</pre>`;
            return;
        }

        content.innerHTML = `<pre>${escapeHtml(currentProject[tab] || "")}</pre>`;
    }

    function escapeHtml(str) {
        return String(str)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;");
    }

    async function loadProjects() {
        const res = await fetch("/api/projects");
        const data = await res.json();

        const box = document.getElementById("projectList");
        box.innerHTML = "";

        data.items.forEach(item => {
            const div = document.createElement("div");
            div.className = "project-item";
            div.innerHTML = `
                <strong>${escapeHtml(item.title)}</strong>
                <div class="small">${escapeHtml(item.raw_idea.slice(0, 50))}</div>
                <div class="small">${escapeHtml(item.created_at)}</div>
            `;
            div.onclick = () => loadProject(item.id);
            box.appendChild(div);
        });
    }

    async function loadProject(id) {
        const res = await fetch(`/api/projects/${id}`);
        currentProject = await res.json();
        showTab(currentTab);
    }

    loadProjects();
</script>
</body>
</html>
""")


# =========================
# 启动方式：
# uvicorn app:app --reload --host 0.0.0.0 --port 8000
# =========================
