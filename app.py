import json
import os
import re
import uuid
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import httpx
import openai
import streamlit as st
import tiktoken
from ollama import Client as OllClient


openai_client = None
ollama_client = None


def calculate_cost(model, tokens_in, tokens_out):
    # 可按需扩展真实计费规则
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


def get_chat_response(api_key, base_url, model, messages, stream=False, temperature=0.7, num_ctx=24000, num_predict=4096):
    global openai_client, ollama_client

    system_prompt = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    prompt = "".join([m["content"] for m in messages if m["role"] == "user"])

    if "ollama" in model:
        parts = re.split('#', model)

        print("parts[1]={}".format(parts[1]))

        if ollama_client is None:
            ollama_client = OllClient(host=base_url, headers={'x-some-header': 'some-value'})
        response = ollama_client.chat(model=parts[1], messages=messages, options={"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict})

        if response:
            answer = response.message.content
        else:
            return None

    else:
        print("Using openai interface api")
        if openai_client is None:
            openai_client = openai.OpenAI(api_key=api_key, base_url=base_url)
        print("api_key:{} base_url:{}".format(api_key, base_url))
        response = None
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                temperature=temperature,
            )
        except httpx.HTTPStatusError as e:
            print("API 请求失败:", e.response.status_code, e.response.text)
            return {"error": "服务器错误，请稍后重试"}
        except Exception as e:
            print(f"Failed to generate chat response. Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            if hasattr(e, 'response'):
                print(f"HTTP Status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")

        if response:
            answer = response.choices[0].message.content
        else:
            return None

    try:
        if model in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1"]:
            encoding = tiktoken.encoding_for_model("gpt-4o")
        elif model == "deepseek-chat":
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.encoding_for_model(model)
    except Exception:
        print("Fallback to default tokenizer")
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens_in = len(encoding.encode(system_prompt + prompt))
    tokens_out = len(encoding.encode(answer))
    _ = calculate_cost(model, tokens_in, tokens_out)

    result_path = save_token_result("default", model, tokens_in, tokens_out, messages, answer)
    print(f"token log saved: {result_path}")
    return answer


DATA_PATH = "data/novel_data.json"


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
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.worlds = []
        for w in raw.get("worlds", []):
            world = World(id=w["id"], name=w["name"], description=w["description"])
            world.characters = [Character(**c) for c in w.get("characters", [])]
            world.scenes = [Scene(**s) for s in w.get("scenes", [])]
            storylines = []
            for sl in w.get("storylines", []):
                nodes = [StoryNode(**n) for n in sl.get("nodes", [])]
                storylines.append(StoryLine(id=sl["id"], name=sl["name"], nodes=nodes))
            world.storylines = storylines
            self.worlds.append(world)

    def save(self):
        obj = {"worlds": [asdict(w) for w in self.worlds]}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)


def llm_json(prompt, api_key, base_url, model):
    messages = [
        {"role": "system", "content": "你是小说策划助手，输出必须是合法 JSON，不要包含代码块标记。"},
        {"role": "user", "content": prompt},
    ]
    res = get_chat_response(api_key, base_url, model, messages)
    if isinstance(res, dict) and res.get("error"):
        raise ValueError(res["error"])
    return json.loads(res)


def make_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def get_world(db: NovelDB, world_id: str) -> Optional[World]:
    return next((w for w in db.worlds if w.id == world_id), None)


def main():
    st.set_page_config(page_title="小说可视化生成器", layout="wide")
    st.title("📚 小说可视化生成器")

    db = NovelDB()

    with st.sidebar:
        st.header("LLM 设置")
        api_key = st.text_input("API Key", type="password")
        base_url = st.text_input("Base URL", value="https://api.openai.com/v1")
        model = st.text_input("Model", value="gpt-4o-mini")
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7)

    tab_world, tab_role, tab_scene, tab_story, tab_export = st.tabs(
        ["世界", "角色", "场景", "故事线", "导出小说"]
    )

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
                if not api_key:
                    st.error("请先填写 API Key")
                else:
                    desc = get_chat_response(
                        api_key,
                        base_url,
                        model,
                        [
                            {"role": "system", "content": "你是小说世界观设计师。"},
                            {"role": "user", "content": f"根据提示词生成世界描述：{prompt}"},
                        ],
                        temperature=temperature,
                    )
            world = World(id=make_id("world"), name=name, description=desc or "")
            db.worlds.append(world)
            db.save()
            st.success("世界已创建")

        if db.worlds:
            selected_world_id = st.selectbox("选择世界", [w.id for w in db.worlds], format_func=lambda x: get_world(db, x).name)
            world = get_world(db, selected_world_id)
            world.description = st.text_area("编辑世界描述", value=world.description, key=f"desc_{world.id}")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("保存世界描述"):
                    db.save()
                    st.success("已保存")
            with col2:
                rewrite_prompt = st.text_input("用提示词改写描述", key=f"rewrite_{world.id}")
                if st.button("LLM 改写") and rewrite_prompt:
                    world.description = get_chat_response(
                        api_key,
                        base_url,
                        model,
                        [
                            {"role": "system", "content": "你是小说世界观润色助手。"},
                            {"role": "user", "content": f"原描述：{world.description}\n修改需求：{rewrite_prompt}"},
                        ],
                        temperature=temperature,
                    )
                    db.save()
                    st.success("已改写并保存")

    world_options = db.worlds
    if not world_options:
        st.info("请先创建至少一个世界。")
        return

    with tab_role:
        st.subheader("角色管理")
        w = st.selectbox("角色所属世界", [x.id for x in world_options], key="role_world", format_func=lambda x: get_world(db, x).name)
        world = get_world(db, w)
        with st.form("add_char"):
            mode = st.radio("创建方式", ["手工", "LLM JSON 生成"], horizontal=True)
            c_name = st.text_input("角色名")
            c_desc = st.text_area("角色描述")
            c_prompt = st.text_area("角色提示词")
            add = st.form_submit_button("新增角色")
        if add:
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

        for c in world.characters:
            with st.expander(f"{c.name} ({c.id})"):
                c.name = st.text_input("名称", c.name, key=f"cn_{c.id}")
                c.description = st.text_area("描述", c.description, key=f"cd_{c.id}")
        if st.button("保存角色修改"):
            db.save()
            st.success("角色已保存")

    with tab_scene:
        st.subheader("场景管理")
        w = st.selectbox("场景所属世界", [x.id for x in world_options], key="scene_world", format_func=lambda x: get_world(db, x).name)
        world = get_world(db, w)
        with st.form("add_scene"):
            mode = st.radio("创建方式", ["手工", "LLM JSON 生成"], horizontal=True, key="scene_mode")
            s_name = st.text_input("场景名")
            s_desc = st.text_area("场景描述")
            s_prompt = st.text_area("场景提示词")
            add = st.form_submit_button("新增场景")
        if add:
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

        for s in world.scenes:
            with st.expander(f"{s.name} ({s.id})"):
                s.name = st.text_input("名称", s.name, key=f"sn_{s.id}")
                s.description = st.text_area("描述", s.description, key=f"sd_{s.id}")
        if st.button("保存场景修改"):
            db.save()
            st.success("场景已保存")

    with tab_story:
        st.subheader("事件发展线与节点")
        w = st.selectbox("故事线所属世界", [x.id for x in world_options], key="story_world", format_func=lambda x: get_world(db, x).name)
        world = get_world(db, w)

        c1, c2 = st.columns([1, 2])
        with c1:
            sl_name = st.text_input("新故事线名称")
            if st.button("创建故事线") and sl_name:
                world.storylines.append(StoryLine(id=make_id("line"), name=sl_name))
                db.save()
                st.success("故事线已创建")

        if not world.storylines:
            st.info("该世界还没有故事线")
        else:
            sid = st.selectbox("选择故事线", [s.id for s in world.storylines], format_func=lambda x: next(i.name for i in world.storylines if i.id == x))
            line = next(i for i in world.storylines if i.id == sid)

            with st.form("add_node"):
                title = st.text_input("节点标题")
                summary = st.text_area("节点故事概要")
                scene_id = st.selectbox("节点场景", [""] + [s.id for s in world.scenes], format_func=lambda x: "无" if x == "" else next(i.name for i in world.scenes if i.id == x))
                char_ids = st.multiselect("节点角色", [c.id for c in world.characters], format_func=lambda x: next(i.name for i in world.characters if i.id == x))
                add_node = st.form_submit_button("新增节点")
            if add_node and title:
                line.nodes.append(StoryNode(id=make_id("node"), title=title, summary=summary, scene_id=scene_id or None, character_ids=char_ids))
                db.save()
                st.success("节点已添加")

            st.markdown("#### 故事线可视化")
            if line.nodes:
                chart = "digraph G { rankdir=LR;"
                for idx, n in enumerate(line.nodes):
                    chart += f'"{n.id}" [label="{idx+1}. {n.title}"];'
                    if idx > 0:
                        chart += f'"{line.nodes[idx-1].id}" -> "{n.id}";'
                chart += "}"
                st.graphviz_chart(chart)

            if st.button("生成该故事线完整小说"):
                history = ""
                for i, n in enumerate(line.nodes, start=1):
                    scene = next((s for s in world.scenes if s.id == n.scene_id), None)
                    chars = [c for c in world.characters if c.id in n.character_ids]
                    msg = [
                        {"role": "system", "content": "你是长篇小说作者，输出中文，多段、多转折、连贯。"},
                        {
                            "role": "user",
                            "content": (
                                f"世界观：{world.description}\n"
                                f"已有剧情脉络：{history or '无'}\n"
                                f"当前节点({i}/{len(line.nodes)}): 标题={n.title} 概要={n.summary}\n"
                                f"场景：{scene.description if scene else '未指定'}\n"
                                f"角色：{'; '.join([f'{c.name}:{c.description}' for c in chars]) or '未指定'}\n"
                                f"请生成该节点小说正文（>=3段），并确保和历史一致。"
                            ),
                        },
                    ]
                    n.generated_text = get_chat_response(api_key, base_url, model, msg, temperature=temperature)
                    history += f"\n[节点{i}:{n.title}] {n.generated_text}\n"
                db.save()
                st.success("已完成所有节点生成")

            for i, n in enumerate(line.nodes, start=1):
                with st.expander(f"{i}. {n.title}"):
                    n.title = st.text_input("标题", n.title, key=f"nt_{n.id}")
                    n.summary = st.text_area("概要", n.summary, key=f"ns_{n.id}")
                    n.generated_text = st.text_area("已生成文本", n.generated_text, height=220, key=f"ng_{n.id}")
            if st.button("保存节点编辑"):
                db.save()
                st.success("节点已保存")

    with tab_export:
        st.subheader("导出完整小说")
        w = st.selectbox("导出世界", [x.id for x in world_options], key="export_world", format_func=lambda x: get_world(db, x).name)
        world = get_world(db, w)
        if not world.storylines:
            st.info("该世界暂无故事线")
        else:
            sid = st.selectbox("选择故事线", [s.id for s in world.storylines], key="export_line", format_func=lambda x: next(i.name for i in world.storylines if i.id == x))
            line = next(i for i in world.storylines if i.id == sid)
            full_text = f"# {world.name} - {line.name}\n\n{world.description}\n\n"
            for idx, n in enumerate(line.nodes, start=1):
                full_text += f"## 第{idx}节 {n.title}\n\n{n.generated_text or '（未生成）'}\n\n"
            st.text_area("预览", full_text, height=360)
            st.download_button("下载 markdown", full_text, file_name=f"{world.name}_{line.name}.md")


if __name__ == "__main__":
    main()
