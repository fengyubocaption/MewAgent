"""面试工具模块 — 简历解析、JD 分析、匹配评估、模拟面试。"""
import json
import logging
import os

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

from backend.db.database import SessionLocal
from backend.db.models import Resume, JobDescription

logger = logging.getLogger(__name__)

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

# ---------- LLM 辅助函数 ----------

def _get_llm(temperature: float = 0.1):
    """获取 LLM 实例（延迟初始化）。"""
    if not API_KEY or not MODEL:
        return None
    return init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=temperature,
    )


def _parse_llm_json(content: str) -> dict:
    """从 LLM 响应中提取 JSON。"""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


# ---------- Prompt 模板 ----------

RESUME_PARSE_PROMPT = """请从以下简历文本中提取结构化信息，返回 JSON 格式。

简历文本：
{text}

输出格式（必须是合法 JSON）：
{{
    "name": "姓名",
    "education": [
        {{"school": "学校", "degree": "学位", "major": "专业", "years": "时间段"}}
    ],
    "experience": [
        {{"company": "公司", "title": "职位", "duration": "时间段", "description": "工作描述"}}
    ],
    "skills": ["技能1", "技能2", ...],
    "projects": [
        {{"name": "项目名", "description": "项目描述", "tech_stack": ["技术1", "技术2"]}}
    ],
    "certifications": ["证书1", ...],
    "summary": "一句话总结（年限+核心能力+求职方向）"
}}

要求：
1. skills 应包含技术栈、框架、工具、编程语言等
2. summary 控制在50字以内
3. 如果某项信息不存在，返回空列表或空字符串
4. 只输出 JSON，不要其他内容"""

JD_PARSE_PROMPT = """请从以下职位描述中提取结构化信息，返回 JSON 格式。

职位描述：
{text}

输出格式（必须是合法 JSON）：
{{
    "title": "职位名称",
    "company": "公司名称",
    "requirements": ["必备要求1", "必备要求2", ...],
    "skills": ["技能1", "技能2", ...],
    "responsibilities": ["职责1", "职责2", ...],
    "nice_to_have": ["加分项1", "加分项2", ...],
    "salary_range": "薪资范围",
    "location": "工作地点"
}}

要求：
1. skills 提取具体的技术栈、工具、框架名称
2. requirements 提取学历、经验年限等硬性要求
3. 如果某项信息不存在，返回空列表或空字符串
4. 只输出 JSON，不要其他内容"""

MATCH_PROMPT = """你是一位资深HR和技术面试官。请分析以下简历与职位描述的匹配度。

## 简历信息
{resume_data}

## 职位要求
{jd_data}

请从以下维度分析并返回 JSON：

{{
    "overall_score": 85,
    "dimension_scores": {{
        "skill_match": {{"score": 90, "detail": "技能匹配度分析"}},
        "experience_match": {{"score": 80, "detail": "经验匹配度分析"}},
        "education_match": {{"score": 85, "detail": "学历匹配度分析"}},
        "project_match": {{"score": 88, "detail": "项目经验匹配度分析"}}
    }},
    "matched_skills": ["匹配的技能1", ...],
    "missing_skills": ["缺失的技能1", ...],
    "strengths": ["优势1", ...],
    "gaps": ["不足1", ...],
    "suggestions": ["建议1", ...],
    "interview_focus": ["面试中可能重点考察的方向1", ...]
}}

评分标准：
- 90+：高度匹配，强烈推荐
- 75-89：较好匹配，建议面试
- 60-74：部分匹配，需要评估
- 60以下：匹配度低

只输出 JSON，不要其他内容。"""

INTERVIEW_QUESTION_PROMPT = """你是一位资深面试官。请根据以下信息生成面试题。

## 简历信息
{resume_data}

## 职位要求
{jd_data}

## 面试类型
{interview_type}

请生成 3 道高质量面试题，返回 JSON：

{{
    "questions": [
        {{
            "id": 1,
            "category": "技术题/行为题/场景题",
            "question": "面试题内容",
            "key_points": ["考察要点1", "考察要点2"],
            "expected_answer_outline": "期望回答的要点概述"
        }}
    ]
}}

面试类型说明：
- 技术面：深入考察技术栈、原理理解、实战经验
- 行为面：使用 STAR 框架，考察软技能、团队协作、问题解决
- 综合面：技术+行为混合，全面评估

要求：
1. 问题要具体，不要泛泛而谈
2. 结合候选人简历中的项目经验出追问
3. 结合 JD 中的核心要求出针对性题目
4. 只输出 JSON，不要其他内容"""

EVALUATE_PROMPT = """你是一位资深面试官。请评估候选人的面试回答。

## 面试题目
{question}

## 候选人回答
{answer}

## 考察要点
{key_points}

请从以下维度评估并返回 JSON：

{{
    "overall_score": 75,
    "dimension_scores": {{
        "technical_accuracy": {{"score": 80, "comment": "技术准确性评价"}},
        "logical_clarity": {{"score": 70, "comment": "逻辑清晰度评价"}},
        "completeness": {{"score": 75, "comment": "回答完整性评价"}},
        "practical_experience": {{"score": 80, "comment": "实战经验体现"}}
    }},
    "strengths": ["回答亮点1", ...],
    "improvements": ["改进建议1", ...],
    "model_answer": "参考答案（简洁版）",
    "follow_up": "建议的追问题目"
}}

评分标准：
- 90+：优秀，回答全面深入
- 75-89：良好，基本到位
- 60-74：一般，需要改进
- 60以下：不足，需要加强

只输出 JSON，不要其他内容。"""


# ---------- 工具函数 ----------

def create_interview_tools(user_id: int):
    """创建面试相关工具。

    Args:
        user_id: 用户 ID

    Returns:
        工具列表：[analyze_resume, analyze_jd, match_resume_jd, mock_interview]
    """

    @tool("analyze_resume")
    def analyze_resume(raw_text: str, filename: str = "resume") -> str:
        """Parse a resume and extract structured information (skills, experience, education, projects).
        Call this when the user uploads a resume or pastes resume text.

        Args:
            raw_text: The full text content of the resume
            filename: The original filename of the resume
        """
        llm = _get_llm()
        if not llm:
            return "错误：LLM 服务未配置"

        if not raw_text or len(raw_text.strip()) < 20:
            return "错误：简历内容过短，请提供完整的简历文本"

        try:
            prompt = RESUME_PARSE_PROMPT.format(text=raw_text[:5000])
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            structured = _parse_llm_json(content)

            if not structured:
                return "错误：无法解析简历内容，请检查格式"

            # 存入数据库
            db = SessionLocal()
            try:
                resume = Resume(
                    user_id=user_id,
                    filename=filename,
                    raw_text=raw_text[:10000],
                    structured_data=structured,
                )
                db.add(resume)
                db.commit()
                db.refresh(resume)
                resume_id = resume.id
            finally:
                db.close()

            # 构建摘要
            name = structured.get("name", "未知")
            skills = structured.get("skills", [])
            summary = structured.get("summary", "")
            edu = structured.get("education", [])
            exp = structured.get("experience", [])

            parts = [f"✅ 简历解析完成（ID: {resume_id}）"]
            parts.append(f"姓名：{name}")
            if summary:
                parts.append(f"摘要：{summary}")
            if skills:
                parts.append(f"技能：{', '.join(skills[:10])}")
            if edu:
                edu_str = " | ".join(f"{e.get('school','')} {e.get('degree','')} {e.get('major','')}" for e in edu[:3])
                parts.append(f"教育：{edu_str}")
            if exp:
                exp_str = " | ".join(f"{e.get('company','')} {e.get('title','')}" for e in exp[:3])
                parts.append(f"经历：{exp_str}")

            return "\n".join(parts)

        except Exception as e:
            logger.error("简历解析失败: %s", e)
            return f"错误：简历解析失败 - {e}"

    @tool("analyze_jd")
    def analyze_jd(jd_text: str, title: str = "", company: str = "") -> str:
        """Analyze a job description and extract structured requirements, skills, and responsibilities.
        Call this when the user pastes a job description.

        Args:
            jd_text: The full text of the job description
            title: Job title (optional, will be extracted if not provided)
            company: Company name (optional, will be extracted if not provided)
        """
        llm = _get_llm()
        if not llm:
            return "错误：LLM 服务未配置"

        if not jd_text or len(jd_text.strip()) < 20:
            return "错误：JD 内容过短，请提供完整的职位描述"

        try:
            prompt = JD_PARSE_PROMPT.format(text=jd_text[:5000])
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            structured = _parse_llm_json(content)

            if not structured:
                return "错误：无法解析 JD 内容，请检查格式"

            # 补充用户提供的标题和公司
            if title and not structured.get("title"):
                structured["title"] = title
            if company and not structured.get("company"):
                structured["company"] = company

            # 存入数据库
            db = SessionLocal()
            try:
                jd = JobDescription(
                    user_id=user_id,
                    title=structured.get("title", title or "未知职位"),
                    company=structured.get("company", company or ""),
                    raw_text=jd_text[:10000],
                    structured_data=structured,
                )
                db.add(jd)
                db.commit()
                db.refresh(jd)
                jd_id = jd.id
            finally:
                db.close()

            # 构建摘要
            parts = [f"✅ JD 分析完成（ID: {jd_id}）"]
            parts.append(f"职位：{structured.get('title', '未知')}")
            if structured.get("company"):
                parts.append(f"公司：{structured['company']}")
            if structured.get("skills"):
                parts.append(f"核心技能：{', '.join(structured['skills'][:10])}")
            if structured.get("requirements"):
                parts.append(f"硬性要求：{'; '.join(structured['requirements'][:5])}")
            if structured.get("nice_to_have"):
                parts.append(f"加分项：{', '.join(structured['nice_to_have'][:5])}")
            if structured.get("salary_range"):
                parts.append(f"薪资：{structured['salary_range']}")
            if structured.get("location"):
                parts.append(f"地点：{structured['location']}")

            return "\n".join(parts)

        except Exception as e:
            logger.error("JD 分析失败: %s", e)
            return f"错误：JD 分析失败 - {e}"

    @tool("match_resume_jd")
    def match_resume_jd(resume_id: int, jd_id: int) -> str:
        """Analyze the match between a resume and a job description.
        Provides match score, matched/missing skills, strengths, gaps, and suggestions.
        Call this when the user asks about resume-JD fit or wants to know their match level.

        Args:
            resume_id: The resume ID (from analyze_resume)
            jd_id: The job description ID (from analyze_jd)
        """
        llm = _get_llm(temperature=0.2)
        if not llm:
            return "错误：LLM 服务未配置"

        db = SessionLocal()
        try:
            resume = db.query(Resume).filter(Resume.id == resume_id, Resume.user_id == user_id).first()
            jd = db.query(JobDescription).filter(JobDescription.id == jd_id, JobDescription.user_id == user_id).first()
        finally:
            db.close()

        if not resume:
            return f"错误：未找到简历（ID: {resume_id}），请先上传简历"
        if not jd:
            return f"错误：未找到 JD（ID: {jd_id}），请先分析 JD"

        try:
            resume_data = json.dumps(resume.structured_data or {}, ensure_ascii=False, indent=2)
            jd_data = json.dumps(jd.structured_data or {}, ensure_ascii=False, indent=2)

            prompt = MATCH_PROMPT.format(resume_data=resume_data, jd_data=jd_data)
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            result = _parse_llm_json(content)

            if not result:
                return "错误：匹配分析失败，无法生成结果"

            # 构建报告
            parts = ["## 📊 简历-JD 匹配分析报告\n"]
            parts.append(f"**综合匹配度：{result.get('overall_score', 0)}/100**\n")

            # 维度评分
            dims = result.get("dimension_scores", {})
            dim_names = {
                "skill_match": "技能匹配",
                "experience_match": "经验匹配",
                "education_match": "学历匹配",
                "project_match": "项目匹配",
            }
            for key, name in dim_names.items():
                if key in dims:
                    d = dims[key]
                    parts.append(f"- {name}：{d.get('score', 0)}/100 — {d.get('detail', '')}")

            # 匹配的技能
            matched = result.get("matched_skills", [])
            if matched:
                parts.append(f"\n### ✅ 匹配的技能\n{', '.join(matched)}")

            # 缺失的技能
            missing = result.get("missing_skills", [])
            if missing:
                parts.append(f"\n### ⚠️ 缺失的技能\n{', '.join(missing)}")

            # 优势
            strengths = result.get("strengths", [])
            if strengths:
                parts.append("\n### 💪 优势")
                for s in strengths:
                    parts.append(f"- {s}")

            # 不足
            gaps = result.get("gaps", [])
            if gaps:
                parts.append("\n### 📌 待提升")
                for g in gaps:
                    parts.append(f"- {g}")

            # 建议
            suggestions = result.get("suggestions", [])
            if suggestions:
                parts.append("\n### 💡 改进建议")
                for s in suggestions:
                    parts.append(f"- {s}")

            # 面试重点
            focus = result.get("interview_focus", [])
            if focus:
                parts.append("\n### 🎯 面试可能重点考察")
                for f in focus:
                    parts.append(f"- {f}")

            return "\n".join(parts)

        except Exception as e:
            logger.error("匹配分析失败: %s", e)
            return f"错误：匹配分析失败 - {e}"

    @tool("mock_interview")
    def mock_interview(jd_id: int, mode: str = "question", interview_type: str = "综合面", answer: str = "", question_text: str = "", key_points: str = "") -> str:
        """Mock interview tool — generate questions or evaluate answers.

        Args:
            jd_id: The job description ID to base the interview on
            mode: "question" to generate interview questions, "evaluate" to evaluate an answer
            interview_type: Type of interview - "技术面" (technical), "行为面" (behavioral), "综合面" (comprehensive)
            answer: The candidate's answer to evaluate (required when mode="evaluate")
            question_text: The interview question being answered (required when mode="evaluate")
            key_points: Key points to evaluate against (optional, for mode="evaluate")
        """
        llm = _get_llm(temperature=0.3)
        if not llm:
            return "错误：LLM 服务未配置"

        db = SessionLocal()
        try:
            jd = db.query(JobDescription).filter(JobDescription.id == jd_id, JobDescription.user_id == user_id).first()
            # 获取最新简历
            resume = db.query(Resume).filter(Resume.user_id == user_id).order_by(Resume.created_at.desc()).first()
        finally:
            db.close()

        if not jd:
            return f"错误：未找到 JD（ID: {jd_id}），请先分析 JD"

        jd_data = json.dumps(jd.structured_data or {}, ensure_ascii=False, indent=2)
        resume_data = json.dumps(resume.structured_data or {}, ensure_ascii=False, indent=2) if resume else "暂无简历信息"

        if mode == "question":
            try:
                prompt = INTERVIEW_QUESTION_PROMPT.format(
                    resume_data=resume_data,
                    jd_data=jd_data,
                    interview_type=interview_type,
                )
                response = llm.invoke(prompt)
                content = response.content if hasattr(response, "content") else str(response)
                result = _parse_llm_json(content)

                if not result or not result.get("questions"):
                    return "错误：面试题生成失败"

                parts = [f"## 🎤 模拟面试 — {interview_type}\n"]
                for q in result["questions"]:
                    parts.append(f"### 题目 {q.get('id', '?')}（{q.get('category', '未知')}）")
                    parts.append(f"**{q.get('question', '')}**\n")
                    if q.get("key_points"):
                        parts.append(f"考察要点：{', '.join(q['key_points'])}")
                    if q.get("expected_answer_outline"):
                        parts.append(f"回答方向：{q['expected_answer_outline']}")
                    parts.append("")

                parts.append("---")
                parts.append("💡 请回答以上题目，回答后我会为你评估。也可以告诉我你想回答第几题。")
                return "\n".join(parts)

            except Exception as e:
                logger.error("面试题生成失败: %s", e)
                return f"错误：面试题生成失败 - {e}"

        elif mode == "evaluate":
            if not answer:
                return "错误：评估模式需要提供回答内容（answer 参数）"
            if not question_text:
                return "错误：评估模式需要提供面试题内容（question_text 参数）"

            try:
                prompt = EVALUATE_PROMPT.format(
                    question=question_text,
                    answer=answer,
                    key_points=key_points or "无特定要点，综合评估",
                )
                response = llm.invoke(prompt)
                content = response.content if hasattr(response, "content") else str(response)
                result = _parse_llm_json(content)

                if not result:
                    return "错误：评估失败"

                parts = ["## 📝 面试回答评估\n"]
                parts.append(f"**综合评分：{result.get('overall_score', 0)}/100**\n")

                dims = result.get("dimension_scores", {})
                dim_names = {
                    "technical_accuracy": "技术准确性",
                    "logical_clarity": "逻辑清晰度",
                    "completeness": "回答完整性",
                    "practical_experience": "实战经验",
                }
                for key, name in dim_names.items():
                    if key in dims:
                        d = dims[key]
                        parts.append(f"- {name}：{d.get('score', 0)}/100 — {d.get('comment', '')}")

                strengths = result.get("strengths", [])
                if strengths:
                    parts.append("\n### ✅ 回答亮点")
                    for s in strengths:
                        parts.append(f"- {s}")

                improvements = result.get("improvements", [])
                if improvements:
                    parts.append("\n### 💡 改进建议")
                    for i in improvements:
                        parts.append(f"- {i}")

                model = result.get("model_answer", "")
                if model:
                    parts.append(f"\n### 📖 参考答案\n{model}")

                follow_up = result.get("follow_up", "")
                if follow_up:
                    parts.append(f"\n### 🔁 追问\n{follow_up}")

                return "\n".join(parts)

            except Exception as e:
                logger.error("回答评估失败: %s", e)
                return f"错误：回答评估失败 - {e}"

        else:
            return f"错误：不支持的模式 '{mode}'，请使用 'question' 或 'evaluate'"

    return [analyze_resume, analyze_jd, match_resume_jd, mock_interview]
