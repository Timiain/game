import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import httpx
import openai
import streamlit as st
import tiktoken
from ollama import Client as OllClient
from streamlit.runtime.scriptrunner import get_script_run_ctx

openai_client = None
ollama_client = None
DATA_PATH = "data/novel_data.json"


def calculate_cost(model, tokens_in, tokens_out):
    return 0.0


def save_token_result(tag, model, tokens_in, tokens_out, messages, answer):
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "token_log.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "tag": tag,
                    "model": model,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "messages": messages,
                    "answer": answer,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return path


def normalize_llm_text(text):
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def get_chat_response(api_key, base_url, model, messages, stream=False, temperature=0.7, num_ctx=24000, num_predict=4096):
    global openai_client, ollama_client

    system_prompt = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    prompt = "".join([m["content"] for m in messages if m["role"] == "user"])

    if "ollama" in model:
        parts = re.split("#", model)
        if len(parts) < 2 or not parts[1].strip():
            return {"error": "ollama 模型格式错误，请使用 ollama#model_name"}
        if ollama_client is None:
            ollama_client = OllClient(host=base_url, headers={"x-some-header": "some-value"})
        response = ollama_client.chat(
            model=parts[1],
            messages=messages,
            options={"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict},
        )
        if not response:
            return None
        answer = response.message.content
    else:
        if not api_key:
            return {"error": "缺少 API Key"}
        if openai_client is None:
            openai_client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = None
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                temperature=temperature,
            )
        except httpx.HTTPStatusError as e:
            return {"error": f"API 请求失败: {e.response.status_code}"}
        except Exception as e:
            return {"error": f"调用失败: {type(e).__name__}: {str(e)}"}

        if not response:
            return None
        answer = response.choices[0].message.content

    try:
        if model in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1"]:
            encoding = tiktoken.encoding_for_model("gpt-4o")
        elif model == "deepseek-chat":
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens_in = len(encoding.encode(system_prompt + prompt))
    tokens_out = len(encoding.encode(answer))
    _ = calculate_cost(model, tokens_in, tokens_out)
    save_token_result("default", model, tokens_in, tokens_out, messages, answer)
    return answer


@dataclass
class Character:
    id: str
    name: str
    description: str


@dataclass
class Scene:
    id: str
    name: str
    description: str


@dataclass
class StoryNode:
    id: str
    title: str
    summary: str
    scene_id: Optional[str] = None
    character_ids: List[str] = field(default_factory=list)
    generated_text: str = ""


@dataclass
class StoryLine:
    id: str
    name: str
    nodes: List[StoryNode] = field(default_factory=list)


@dataclass
class World:
    id: str
    name: str
    description: str
    characters: List[Character] = field(default_factory=list)
    scenes: List[Scene] = field(default_factory=list)
    storylines: List[StoryLine] = field(default_factory=list)


class NovelDB:
    def __init__(self, path=DATA_PATH):
        self.path = path
        self.worlds: List[World] = []
        self.load()

    def load(self):
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.path):
            self.save()
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except json.JSONDecodeError:
            raw = {"worlds": []}

        self.worlds = []
        for w in raw.get("worlds", []):
            world = World(id=w["id"], name=w["name"], description=w.get("description", ""))
            world.characters = [Character(**c) for c in w.get("characters", [])]
            world.scenes = [Scene(**s) for s in w.get("scenes", [])]
            world.storylines = [
                StoryLine(
                    id=sl["id"],
                    name=sl["name"],
                    nodes=[StoryNode(**n) for n in sl.get("nodes", [])],
                )
                for sl in w.get("storylines", [])
            ]
            self.worlds.append(world)

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"worlds": [asdict(w) for w in self.worlds]}, f, ensure_ascii=False, indent=2)


def llm_json(prompt, api_key, base_url, model):
    messages = [
        {"role": "system", "content": "你是小说策划助手，输出必须是合法 JSON，不要包含代码块标记。"},
        {"role": "user", "content": prompt},
    ]
    res = get_chat_response(api_key, base_url, model, messages)
    if isinstance(res, dict) and res.get("error"):
        raise ValueError(res["error"])
    return json.loads(normalize_llm_text(res))


def generate_node_text(world, line, node, idx, total, history, llm_cfg):
    scene = next((s for s in world.scenes if s.id == node.scene_id), None)
    chars = [c for c in world.characters if c.id in node.character_ids]
    msg = [
        {"role": "system", "content": "你是长篇小说作者，输出中文，多段、多转折、连贯。"},
        {
            "role": "user",
            "content": (
                f"世界名：{world.name}\n"
                f"世界观：{world.description}\n"
                f"故事线：{line.name}\n"
                f"已有剧情脉络：{history or '无'}\n"
                f"当前节点({idx}/{total}): 标题={node.title} 概要={node.summary}\n"
                f"场景：{scene.description if scene else '未指定'}\n"
                f"角色：{'; '.join([f'{c.name}:{c.description}' for c in chars]) or '未指定'}\n"
                "请生成该节点小说正文（至少3段，每段有推进，包含冲突或转折，结尾给下一节点留钩子）。"
            ),
        },
    ]
    return get_chat_response(
        llm_cfg["api_key"],
        llm_cfg["base_url"],
        llm_cfg["model"],
        msg,
        temperature=llm_cfg["temperature"],
        num_ctx=llm_cfg["num_ctx"],
        num_predict=llm_cfg["num_predict"],
    )


def render_generation_animation(current_idx, total, node_title, stage_text):
    progress = current_idx / max(total, 1)
    st.progress(progress, text=f"生成进度：{current_idx}/{total}")
    st.caption(f"🎬 正在生成节点《{node_title}》- {stage_text}")


def typewriter_text(container, text, speed=0.01, chunk_size=18):
    if not text:
        container.info("该节点暂未生成文本。")
        return
    shown = ""
    for i in range(0, len(text), chunk_size):
        shown += text[i : i + chunk_size]
        container.markdown(shown)
        time.sleep(speed)


def make_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def get_world(db, world_id):
    return next((w for w in db.worlds if w.id == world_id), None)


def main():
    st.set_page_config(page_title="小说可视化生成器", layout="wide")
    st.title("📚 小说可视化生成器")
    st.caption("面向策划与写作团队：从世界观到可导出的连贯长篇，一站式可视化编排。")

    db = NovelDB()

    with st.sidebar:
        st.header("LLM 设置")
        api_key = st.text_input("API Key", type="password")
        base_url = st.text_input("Base URL", value="https://api.openai.com/v1")
        model = st.text_input("Model", value="gpt-4o-mini")
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7)
        num_ctx = st.number_input("num_ctx", min_value=1024, value=24000, step=1024)
        num_predict = st.number_input("num_predict", min_value=256, value=4096, step=256)

    llm_cfg = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "temperature": temperature,
        "num_ctx": int(num_ctx),
        "num_predict": int(num_predict),
    }

    if "play_animation" not in st.session_state:
        st.session_state.play_animation = True
    if "last_generation_report" not in st.session_state:
        st.session_state.last_generation_report = []

    with st.expander("🚀 使用引导（推荐）", expanded=False):
        st.markdown(
            "1. 先在【世界】中建立世界观。\n"
            "2. 在【角色】【场景】补齐素材资产。\n"
            "3. 在【故事线】配置节点、关联人物/场景并生成。\n"
            "4. 在【导出小说】统一预览与下载。"
        )
        st.session_state.play_animation = st.toggle(
            "启用生成过程动画", value=st.session_state.play_animation
        )

    world_count = len(db.worlds)
    char_count = sum(len(w.characters) for w in db.worlds)
    scene_count = sum(len(w.scenes) for w in db.worlds)
    line_count = sum(len(w.storylines) for w in db.worlds)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("世界", world_count)
    m2.metric("角色", char_count)
    m3.metric("场景", scene_count)
    m4.metric("故事线", line_count)

    tab_world, tab_role, tab_scene, tab_story, tab_export = st.tabs(["世界", "角色", "场景", "故事线", "导出小说"])

    with tab_world:
        st.subheader("世界管理")
        with st.form("new_world"):
            name = st.text_input("世界名称")
            prompt = st.text_area("世界提示词（可选，点击生成将调用 LLM）")
            manual_desc = st.text_area("世界描述（可手工输入）")
            create = st.form_submit_button("创建世界")

        if create and name:
            desc = manual_desc
            if prompt:
                res = get_chat_response(
                    api_key,
                    base_url,
                    model,
                    [
                        {"role": "system", "content": "你是小说世界观设计师。"},
                        {"role": "user", "content": f"根据提示词生成世界描述：{prompt}"},
                    ],
                    temperature=temperature,
                )
                if isinstance(res, dict) and res.get("error"):
                    st.error(res["error"])
                    return
                desc = res
            db.worlds.append(World(id=make_id("world"), name=name, description=desc or ""))
            db.save()
            st.success("世界已创建")

        if db.worlds:
            selected_world_id = st.selectbox("选择世界", [w.id for w in db.worlds], format_func=lambda x: get_world(db, x).name)
            world = get_world(db, selected_world_id)
            world.description = st.text_area("编辑世界描述", value=world.description, key=f"desc_{world.id}")

            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                if st.button("保存世界描述"):
                    db.save()
                    st.success("已保存")
            with c2:
                rewrite_prompt = st.text_input("用提示词改写描述", key=f"rewrite_{world.id}")
            with c3:
                if st.button("LLM 改写") and rewrite_prompt:
                    res = get_chat_response(
                        api_key,
                        base_url,
                        model,
                        [
                            {"role": "system", "content": "你是小说世界观润色助手。"},
                            {"role": "user", "content": f"原描述：{world.description}\n修改需求：{rewrite_prompt}"},
                        ],
                        temperature=temperature,
                    )
                    if isinstance(res, dict) and res.get("error"):
                        st.error(res["error"])
                    else:
                        world.description = res
                        db.save()
                        st.success("已改写并保存")

            if st.button("删除当前世界", type="secondary"):
                db.worlds = [w for w in db.worlds if w.id != world.id]
                db.save()
                st.rerun()

    if not db.worlds:
        st.info("请先创建至少一个世界。")
        return

    with tab_role:
        st.subheader("角色管理")
        world = get_world(
            db,
            st.selectbox("角色所属世界", [x.id for x in db.worlds], key="role_world", format_func=lambda x: get_world(db, x).name),
        )
        with st.form("add_char"):
            mode = st.radio("创建方式", ["手工", "LLM JSON 生成"], horizontal=True)
            c_name = st.text_input("角色名")
            c_desc = st.text_area("角色描述")
            c_prompt = st.text_area("角色提示词")
            add = st.form_submit_button("新增角色")

        if add:
            try:
                if mode == "LLM JSON 生成":
                    payload = llm_json(
                        f"根据提示词生成角色 JSON，格式: {{\"name\":\"\",\"description\":\"\"}}。提示词：{c_prompt}",
                        api_key,
                        base_url,
                        model,
                    )
                    world.characters.append(Character(id=make_id("char"), name=payload["name"], description=payload["description"]))
                else:
                    world.characters.append(Character(id=make_id("char"), name=c_name, description=c_desc))
                db.save()
                st.success("角色已添加")
            except Exception as e:
                st.error(f"角色生成失败: {e}")

        for c in world.characters:
            with st.expander(f"{c.name} ({c.id})"):
                c.name = st.text_input("名称", c.name, key=f"cn_{c.id}")
                c.description = st.text_area("描述", c.description, key=f"cd_{c.id}")
                if st.button("删除该角色", key=f"del_char_{c.id}"):
                    world.characters = [x for x in world.characters if x.id != c.id]
                    for line in world.storylines:
                        for n in line.nodes:
                            n.character_ids = [cid for cid in n.character_ids if cid != c.id]
                    db.save()
                    st.rerun()
        if st.button("保存角色修改"):
            db.save()
            st.success("角色已保存")

    with tab_scene:
        st.subheader("场景管理")
        world = get_world(
            db,
            st.selectbox("场景所属世界", [x.id for x in db.worlds], key="scene_world", format_func=lambda x: get_world(db, x).name),
        )
        with st.form("add_scene"):
            mode = st.radio("创建方式", ["手工", "LLM JSON 生成"], horizontal=True, key="scene_mode")
            s_name = st.text_input("场景名")
            s_desc = st.text_area("场景描述")
            s_prompt = st.text_area("场景提示词")
            add = st.form_submit_button("新增场景")
        if add:
            try:
                if mode == "LLM JSON 生成":
                    payload = llm_json(
                        f"根据提示词生成场景 JSON，格式: {{\"name\":\"\",\"description\":\"\"}}。提示词：{s_prompt}",
                        api_key,
                        base_url,
                        model,
                    )
                    world.scenes.append(Scene(id=make_id("scene"), name=payload["name"], description=payload["description"]))
                else:
                    world.scenes.append(Scene(id=make_id("scene"), name=s_name, description=s_desc))
                db.save()
                st.success("场景已添加")
            except Exception as e:
                st.error(f"场景生成失败: {e}")

        for s in world.scenes:
            with st.expander(f"{s.name} ({s.id})"):
                s.name = st.text_input("名称", s.name, key=f"sn_{s.id}")
                s.description = st.text_area("描述", s.description, key=f"sd_{s.id}")
                if st.button("删除该场景", key=f"del_scene_{s.id}"):
                    world.scenes = [x for x in world.scenes if x.id != s.id]
                    for line in world.storylines:
                        for n in line.nodes:
                            if n.scene_id == s.id:
                                n.scene_id = None
                    db.save()
                    st.rerun()
        if st.button("保存场景修改"):
            db.save()
            st.success("场景已保存")

    with tab_story:
        st.subheader("事件发展线与节点")
        world = get_world(
            db,
            st.selectbox("故事线所属世界", [x.id for x in db.worlds], key="story_world", format_func=lambda x: get_world(db, x).name),
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            sl_name = st.text_input("新故事线名称")
        with c2:
            if st.button("创建故事线") and sl_name:
                world.storylines.append(StoryLine(id=make_id("line"), name=sl_name))
                db.save()
                st.rerun()

        if not world.storylines:
            st.info("该世界还没有故事线")
        else:
            line = next(
                i
                for i in world.storylines
                if i.id
                == st.selectbox("选择故事线", [s.id for s in world.storylines], format_func=lambda x: next(i.name for i in world.storylines if i.id == x))
            )

            c1, c2 = st.columns([4, 1])
            with c1:
                line.name = st.text_input("故事线名称", line.name)
            with c2:
                if st.button("删除故事线"):
                    world.storylines = [x for x in world.storylines if x.id != line.id]
                    db.save()
                    st.rerun()

            with st.form("add_node"):
                title = st.text_input("节点标题")
                summary = st.text_area("节点故事概要")
                scene_id = st.selectbox("节点场景", [""] + [s.id for s in world.scenes], format_func=lambda x: "无" if x == "" else next(i.name for i in world.scenes if i.id == x))
                char_ids = st.multiselect("节点角色", [c.id for c in world.characters], format_func=lambda x: next(i.name for i in world.characters if i.id == x))
                add_node = st.form_submit_button("新增节点")
            if add_node and title:
                line.nodes.append(StoryNode(id=make_id("node"), title=title, summary=summary, scene_id=scene_id or None, character_ids=char_ids))
                db.save()
                st.rerun()

            st.markdown("#### 故事线可视化")
            if line.nodes:
                chart = "digraph G { rankdir=LR;"
                for idx, n in enumerate(line.nodes):
                    safe_label = n.title.replace('"', "'")
                    chart += f'"{n.id}" [label="{idx+1}. {safe_label}"];'
                    if idx > 0:
                        chart += f'"{line.nodes[idx-1].id}" -> "{n.id}";'
                chart += "}"
                st.graphviz_chart(chart)

            st.markdown("#### 生成控制台")
            st.caption("支持全量生成与单节点重生成；可开启动画查看生成过程。")

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("生成该故事线完整小说"):
                    history = ""
                    st.session_state.last_generation_report = []
                    progress_holder = st.empty()
                    stage_holder = st.empty()
                    preview_holder = st.container(border=True)

                    for i, n in enumerate(line.nodes, start=1):
                        if st.session_state.play_animation:
                            with progress_holder:
                                render_generation_animation(i, len(line.nodes), n.title, "构建上下文")
                            time.sleep(0.15)

                        with stage_holder:
                            st.info(f"🧠 生成节点 {i}/{len(line.nodes)}：{n.title}")

                        res = generate_node_text(world, line, n, i, len(line.nodes), history, llm_cfg)
                        if isinstance(res, dict) and res.get("error"):
                            st.error(f"第{i}节点生成失败: {res['error']}")
                            st.session_state.last_generation_report.append(f"❌ 节点{i} {n.title} 失败")
                            break

                        n.generated_text = res
                        history += f"\n[节点{i}:{n.title}] {n.generated_text}\n"
                        st.session_state.last_generation_report.append(f"✅ 节点{i} {n.title} 完成")

                        with preview_holder:
                            st.markdown(f"**最新完成节点：{n.title}**")
                            if st.session_state.play_animation:
                                ph = st.empty()
                                typewriter_text(ph, n.generated_text[:320], speed=0.004, chunk_size=24)
                            else:
                                short = n.generated_text[:320]
                                st.write(short + ("..." if len(n.generated_text) > 320 else ""))

                    db.save()
                    st.success("已完成所有可生成节点")

            with c2:
                node_target = st.selectbox("单节点重生成", [n.id for n in line.nodes] if line.nodes else [], format_func=lambda x: next(n.title for n in line.nodes if n.id == x) if x else x)
                if line.nodes and st.button("仅重生成选中节点"):
                    idx = [n.id for n in line.nodes].index(node_target)
                    history = ""
                    for p, n in enumerate(line.nodes[:idx], start=1):
                        history += f"\n[节点{p}:{n.title}] {n.generated_text}\n"
                    target_node = line.nodes[idx]
                    anim_holder = st.empty()
                    if st.session_state.play_animation:
                        with anim_holder:
                            render_generation_animation(idx + 1, len(line.nodes), target_node.title, "单节点重生成")
                        time.sleep(0.2)

                    res = generate_node_text(world, line, target_node, idx + 1, len(line.nodes), history, llm_cfg)
                    if isinstance(res, dict) and res.get("error"):
                        st.error(res["error"])
                    else:
                        target_node.generated_text = res
                        db.save()
                        st.success("节点重生成成功")

            if st.session_state.last_generation_report:
                with st.expander("📜 最近一次生成报告", expanded=True):
                    for item in st.session_state.last_generation_report:
                        st.write(item)


            for i, n in enumerate(line.nodes, start=1):
                with st.expander(f"{i}. {n.title}"):
                    n.title = st.text_input("标题", n.title, key=f"nt_{n.id}")
                    n.summary = st.text_area("概要", n.summary, key=f"ns_{n.id}")
                    n.scene_id = st.selectbox("场景", [""] + [s.id for s in world.scenes], index=([""] + [s.id for s in world.scenes]).index(n.scene_id) if n.scene_id in [s.id for s in world.scenes] else 0, key=f"nsc_{n.id}", format_func=lambda x: "无" if x == "" else next(i.name for i in world.scenes if i.id == x)) or None
                    n.character_ids = st.multiselect("角色", [c.id for c in world.characters], default=[cid for cid in n.character_ids if cid in [c.id for c in world.characters]], key=f"nch_{n.id}", format_func=lambda x: next(i.name for i in world.characters if i.id == x))
                    n.generated_text = st.text_area("已生成文本", n.generated_text, height=220, key=f"ng_{n.id}")
                    cc1, cc2, cc3 = st.columns([1, 1, 1])
                    with cc1:
                        if st.button("上移", key=f"up_{n.id}") and i > 1:
                            line.nodes[i - 2], line.nodes[i - 1] = line.nodes[i - 1], line.nodes[i - 2]
                            db.save()
                            st.rerun()
                    with cc2:
                        if st.button("下移", key=f"down_{n.id}") and i < len(line.nodes):
                            line.nodes[i - 1], line.nodes[i] = line.nodes[i], line.nodes[i - 1]
                            db.save()
                            st.rerun()
                    with cc3:
                        if st.button("删除节点", key=f"del_node_{n.id}"):
                            line.nodes = [x for x in line.nodes if x.id != n.id]
                            db.save()
                            st.rerun()
            if st.button("保存节点编辑"):
                db.save()
                st.success("节点已保存")

    with tab_export:
        st.subheader("导出完整小说")
        world = get_world(
            db,
            st.selectbox("导出世界", [x.id for x in db.worlds], key="export_world", format_func=lambda x: get_world(db, x).name),
        )
        if not world.storylines:
            st.info("该世界暂无故事线")
        else:
            line = next(i for i in world.storylines if i.id == st.selectbox("选择故事线", [s.id for s in world.storylines], key="export_line", format_func=lambda x: next(i.name for i in world.storylines if i.id == x)))
            full_text = f"# {world.name} - {line.name}\n\n{world.description}\n\n"
            for idx, n in enumerate(line.nodes, start=1):
                full_text += f"## 第{idx}节 {n.title}\n\n{n.generated_text or '（未生成）'}\n\n"
            st.text_area("预览", full_text, height=360)
            st.download_button("下载 markdown", full_text, file_name=f"{world.name}_{line.name}.md")


if __name__ == "__main__":
    if get_script_run_ctx() is None:
        print("请使用以下命令启动应用：streamlit run app.py")
    else:
        main()
