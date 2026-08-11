# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 68 — картинки и голос НА ВХОД, график и таблица НА ВЫХОД.

Наблюдение с прогона UI: у окна ассистента не было ни вставки скриншота, ни
голоса, а «вывод песочницы» выглядел чисто текстовым — при том что matplotlib
в песочнице есть. Разбор показал ТРИ разные причины, и лечатся они по-разному:

1. **вход картинки** — сессия была строго текстовой (`Message.content: str`),
   поэтому мультимодальный запрос собрать было нечем. Решение: в сессии живут
   ССЫЛКИ (`Message.images` → sha256 вложений), а data-URL собирается на момент
   отправки. Складывать base64 в `session.json` нельзя: оценка бюджета
   контекста считает символы, и один скриншот «съел» бы окно модели;
2. **голос** — распознаём ОТДЕЛЬНЫМ эндпоинтом (`/audio/transcriptions`), а не
   аудио прямо в чат: рабочая модель разговора держится на tool-calling и
   длинном контексте, аудио на вход принимают другие. Текст фразы остаётся в
   сессии — его видно, можно поправить и переспросить;
3. **выход песочницы** — файлы создавались во ВРЕМЕННОМ workdir и умирали в
   `close()`. Плюс найден врущий признак: matplotlib кешировал шрифты в
   `~/.matplotlib`, сторож это запрещал, и УСПЕШНЫЙ прогон (rc=0, график
   сохранён) помечался `denied='write'`.

По канону `.clinerules` тесты проверяют ЧИСТУЮ логику без запуска Streamlit;
сеть не трогается (транспорт подменяется), а песочница гоняется настоящая —
иначе ложный `denied` не был бы виден.
"""
import base64
import json
import os

import pytest

from src.assistant import files as afiles
from src.assistant import llm, views
from src.assistant.context import build_turn_messages, run_turn
from src.assistant.llm import LLMError
from src.assistant.session import Artifact, new_session

PROJECT = "pvc_edge_v1"

#: Минимальный валидный PNG 1×1 (чтобы не тащить Pillow в тест).
PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8"
    "AAAwAB/AL+g2sAAAAASUVORK5CYII=")


def _session():
    s = new_session(PROJECT)
    s.web_enabled = False
    return s


class _Ctx:
    """Минимальный контекст инструментов (движка кампании тут не нужно)."""

    def __init__(self, root, project=PROJECT):
        self.root = str(root)
        self.project = project
        self.runner = None
        self.session = None
        self.extra = {}


# ----------------------------------------------------------------------
# 1. Вложение-картинка принимается и НЕ выглядит «непрочитанной»
# ----------------------------------------------------------------------
def test_image_is_accepted_as_attachment(tmp_path):
    s = _session()
    att = afiles.attach_file(s, tmp_path, "screen.png", PNG_1PX, project=PROJECT)
    assert att.mime == "image/png"
    assert att.text == ""                      # текста нет ПО ЗАМЫСЛУ
    assert "изображение" in att.note           # ...и это сказано словами
    assert (tmp_path / PROJECT / "assistant" / "files" / att.stored_name).exists()


def test_image_row_says_picture_not_zero_chars(tmp_path):
    """0 символов читалось бы как «файл не прочитался» — пишем «картинка»."""
    s = _session()
    afiles.attach_file(s, tmp_path, "screen.png", PNG_1PX, project=PROJECT)
    row = views.attachments_dataframe(s).iloc[0]
    assert row["символов"] == "— (картинка)"


def test_oversized_image_refused_with_reason(tmp_path):
    s = _session()
    big = PNG_1PX + b"\x00" * (afiles.MAX_IMAGE_BYTES + 1)
    with pytest.raises(afiles.AttachmentError) as exc:
        afiles.attach_file(s, tmp_path, "huge.png", big, project=PROJECT)
    assert "уходит в запрос" in str(exc.value)
    assert s.attachments == []


def test_unsupported_image_format_lists_supported(tmp_path):
    s = _session()
    with pytest.raises(afiles.AttachmentError) as exc:
        afiles.attach_file(s, tmp_path, "photo.heic", PNG_1PX, project=PROJECT)
    assert ".png" in str(exc.value)            # видно, что принимаем


# ----------------------------------------------------------------------
# 2. Картинка едет в запрос как data-URL, а в сессии остаётся ссылка
# ----------------------------------------------------------------------
def test_data_url_is_built_from_disk(tmp_path):
    s = _session()
    att = afiles.attach_file(s, tmp_path, "screen.png", PNG_1PX, project=PROJECT)
    url = afiles.attachment_data_url(s, tmp_path, att.sha256, project=PROJECT)
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == PNG_1PX


def test_text_attachment_is_not_sendable_as_image(tmp_path):
    s = _session()
    afiles.attach_file(s, tmp_path, "tds.txt", b"d50 = 2.5", project=PROJECT)
    with pytest.raises(afiles.AttachmentError, match="не изображение"):
        afiles.attachment_data_url(s, tmp_path, "tds.txt", project=PROJECT)


def test_user_content_stays_a_plain_string_without_images():
    """Текстовый путь не должен измениться: строка, а не список из одной части."""
    assert llm.user_content("почему такая граница") == "почему такая граница"


def test_user_content_puts_text_before_images():
    parts = llm.user_content("что на скриншоте", ["data:image/png;base64,AAA"])
    assert [p["type"] for p in parts] == ["text", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png")


def test_session_keeps_reference_not_base64(tmp_path):
    """base64 в session.json раздул бы и файл, и оценку бюджета контекста."""
    s = _session()
    att = afiles.attach_file(s, tmp_path, "screen.png", PNG_1PX, project=PROJECT)
    msg = s.add_message("user", "что тут", images=[att.sha256])
    assert msg.images == [att.sha256]
    dumped = json.dumps(msg.to_state(), ensure_ascii=False)
    assert "base64" not in dumped
    assert msg.chat_message()["content"] == "что тут"   # строка, не список


def test_images_attach_to_the_question_in_the_request(tmp_path):
    s = _session()
    msgs = build_turn_messages(s, question="что на графике",
                               image_urls=["data:image/png;base64,AAA"])
    last = msgs[-1]
    assert last["role"] == "user" and llm.has_images(last)
    assert last["content"][0]["text"] == "что на графике"


# ----------------------------------------------------------------------
# 3. Ход с картинкой: запрос мультимодальный, отказ — объяснённый
# ----------------------------------------------------------------------
def _ok_transport(text="Вижу график.", seen=None):
    def transport(payload, *, key="", timeout=0, url=""):
        if seen is not None:
            seen["payload"] = payload
        return {"choices": [{"message": {"role": "assistant",
                                         "content": text}}]}
    return transport


def test_run_turn_sends_image_and_records_reference(tmp_path):
    s = _session()
    att = afiles.attach_file(s, tmp_path, "screen.png", PNG_1PX, project=PROJECT)
    seen = {}
    res = run_turn(s, _Ctx(tmp_path), "что на скриншоте", images=[att.sha256],
                   transport=_ok_transport(seen=seen))
    assert res.ok and res.images == [att.sha256] and not res.image_errors
    user_msgs = [m for m in seen["payload"]["messages"] if m["role"] == "user"]
    assert any(llm.has_images(m) for m in user_msgs)
    assert s.messages[0].images == [att.sha256]


def test_missing_image_is_reported_not_swallowed(tmp_path):
    """Пропавший файл обязан быть назван: иначе «не вижу картинку» необъясним."""
    s = _session()
    att = afiles.attach_file(s, tmp_path, "screen.png", PNG_1PX, project=PROJECT)
    (tmp_path / PROJECT / "assistant" / "files" / att.stored_name).unlink()
    seen = {}
    res = run_turn(s, _Ctx(tmp_path), "что на скриншоте", images=[att.sha256],
                   transport=_ok_transport(text="—", seen=seen))
    assert res.image_errors and "screen.png" in res.image_errors[0]
    sys_texts = " ".join(str(m["content"]) for m in seen["payload"]["messages"]
                         if m["role"] == "system")
    assert "НЕ УДАЛОСЬ приложить изображения" in sys_texts


def test_image_without_question_gets_an_explicit_question(tmp_path):
    s = _session()
    att = afiles.attach_file(s, tmp_path, "screen.png", PNG_1PX, project=PROJECT)
    res = run_turn(s, _Ctx(tmp_path), "", images=[att.sha256],
                   transport=_ok_transport(text="ок"))
    assert res.ok and "изображение" in res.question


def test_turn_without_question_and_without_image_is_refused(tmp_path):
    with pytest.raises(ValueError):
        run_turn(_session(), _Ctx(tmp_path), "   ")


def test_model_without_vision_gets_a_human_hint():
    """HTTP 400 про модальность объясняется словами, а не сырым JSON."""
    assert llm._looks_like_modality_error(
        '{"error":{"message":"This model does not support image input"}}')
    assert not llm._looks_like_modality_error('{"error":"bad json"}')


# ----------------------------------------------------------------------
# 4. Голос: распознаётся ОТДЕЛЬНО и приходит в диалог текстом
# ----------------------------------------------------------------------
WAV = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 32


def test_transcribe_sends_raw_base64_to_the_stt_endpoint():
    """Провайдер ждёт «сырой» base64 БЕЗ префикса data: — на этом легко ошибиться."""
    seen = {}

    def transport(payload, *, key="", timeout=0, url=""):
        seen.update(payload=payload, url=url)
        return {"text": "почему у DINP такая граница",
                "usage": {"seconds": 2, "cost": 0.0001}}

    out = llm.transcribe(WAV, fmt="wav", key="k", transport=transport)
    assert out["text"] == "почему у DINP такая граница"
    assert seen["url"] == llm.TRANSCRIBE_URL
    audio = seen["payload"]["input_audio"]
    assert audio["format"] == "wav"
    assert not audio["data"].startswith("data:")
    assert base64.b64decode(audio["data"]) == WAV


def test_empty_transcription_is_an_explicit_error():
    """Пустой текст нельзя отправить в диалог молча: «не понял» необъясним."""
    def transport(payload, *, key="", timeout=0, url=""):
        return {"text": "   "}

    with pytest.raises(LLMError, match="не распознана"):
        llm.transcribe(WAV, key="k", transport=transport)


def test_empty_audio_refused():
    with pytest.raises(LLMError, match="Пустая аудиозапись"):
        llm.transcribe(b"", key="k", transport=lambda *a, **kw: {})


def test_too_long_audio_refused_with_provider_timeout_reason():
    big = b"\x00" * (llm.MAX_AUDIO_BYTES + 1)
    with pytest.raises(LLMError, match="60 с"):
        llm.transcribe(big, key="k", transport=lambda *a, **kw: {})


def test_stt_model_is_overridable(monkeypatch):
    monkeypatch.delenv("DOE_ASSISTANT_STT", raising=False)
    assert llm.stt_model() == llm.DEFAULT_STT_MODEL
    monkeypatch.setenv("DOE_ASSISTANT_STT", "openai/gpt-4o-mini-transcribe")
    assert llm.stt_model() == "openai/gpt-4o-mini-transcribe"


def test_recognized_text_goes_through_the_normal_pipeline(tmp_path):
    """Смысл выбора STT: распознанная фраза — обычный вопрос, с инструментами."""
    s = _session()
    seen = {}
    res = run_turn(s, _Ctx(tmp_path), "объясни ось DINP",
                   transport=_ok_transport(text="Отвечаю.", seen=seen))
    assert res.ok and s.messages[0].content == "объясни ось DINP"
    assert isinstance(seen["payload"]["messages"][-1]["content"], str)


# ----------------------------------------------------------------------
# 5. Выход песочницы: график и таблица доезжают до кампании
# ----------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


@pytest.fixture
def sandbox(tmp_path):
    from src.assistant.sandbox import SandboxPolicy, SubprocessSandbox
    sb = SubprocessSandbox(SandboxPolicy(repo_root=REPO_ROOT,
                                         workdir=str(tmp_path / "work"),
                                         timeout_s=180.0))
    yield sb
    sb.close()


def _sandbox_ctx(tmp_path, sb, session):
    ctx = _Ctx(tmp_path)
    ctx.session = session
    ctx.extra = {"sandbox": sb}
    return ctx


def test_output_kind_by_suffix():
    from src.assistant.sandbox import output_kind
    assert output_kind("curve.png") == "image"
    assert output_kind("table.csv") == "table"
    assert output_kind("dump.log") == "text"


def test_new_files_ignores_what_was_there_before(sandbox):
    """Отличаем СОЗДАННОЕ прогоном от скрипта и кешей, лежавших раньше."""
    sandbox.write_scratch("old.csv", "a,b\n1,2\n")
    before = sandbox.snapshot_workdir()
    sandbox.write_scratch("fresh.csv", "a,b\n3,4\n")
    produced = [os.path.basename(p) for p in sandbox.new_files(before)]
    assert "fresh.csv" in produced and "old.csv" not in produced


def test_matplotlib_run_is_not_falsely_denied(tmp_path, sandbox):
    """Ложный `denied='write'` из-за кеша шрифтов ~/.matplotlib (регресс).

    Раньше УСПЕШНЫЙ прогон с сохранённым графиком помечался отказом записи:
    инструмент сообщал о запрете, которого по сути не было. Кеш живёт в
    workdir (`MPLCONFIGDIR`), поэтому причины больше нет.
    """
    from src.assistant.tools.sandbox_tools import run_python
    s = _session()
    ctx = _sandbox_ctx(tmp_path, sandbox, s)
    out = run_python(ctx, "import matplotlib\n"
                          "matplotlib.use('Agg')\n"
                          "import matplotlib.pyplot as plt\n"
                          "plt.plot([0, 1], [0, 1])\n"
                          "plt.savefig('curve.png')\n"
                          "print('готово')\n", timeout_s=180)
    assert out["denied"] == "" and out["ok"], out["stderr"][:400]
    assert "готово" in out["stdout"]


def test_plot_and_table_are_collected_into_the_campaign(tmp_path, sandbox):
    """Файлы переезжают из временного workdir в проект — иначе они умирают."""
    from src.assistant.tools.sandbox_tools import run_python
    s = _session()
    ctx = _sandbox_ctx(tmp_path, sandbox, s)
    out = run_python(ctx, "import matplotlib\n"
                          "matplotlib.use('Agg')\n"
                          "import matplotlib.pyplot as plt\n"
                          "import pandas as pd\n"
                          "plt.plot([0, 1, 2], [0, 1, 4])\n"
                          "plt.savefig('curve.png')\n"
                          "pd.DataFrame({'phr': [4, 9]}).to_csv('t.csv',"
                          " index=False)\n", timeout_s=180)
    kinds = {f["kind"] for f in out["outputs"]}
    assert kinds == {"image", "table"}
    for f in out["outputs"]:
        assert os.path.exists(f["path"])       # лежит в кампании, а не в temp
        assert str(tmp_path) in f["path"]
    assert "УЖЕ ПОКАЗАНЫ пользователю" in out["outputs_note"]
    assert {a.kind for a in s.artifacts} == {"text", "image", "table"}


def test_scratch_script_is_not_collected_as_output(tmp_path, sandbox):
    """`snippet.py` — наш же исходник, в выхлоп ему не место."""
    from src.assistant.tools.sandbox_tools import run_python
    s = _session()
    ctx = _sandbox_ctx(tmp_path, sandbox, s)
    out = run_python(ctx, "print('без файлов')", timeout_s=120)
    assert out["outputs"] == []
    assert [a.kind for a in s.artifacts] == ["text"]   # только лог прогона


# ----------------------------------------------------------------------
# 6. Показ: чем рисовать и что НЕ показывать
# ----------------------------------------------------------------------
def _artifact(tmp_path, name, kind, data=b"x"):
    path = tmp_path / name
    path.write_bytes(data)
    return Artifact(name=name, kind=kind, path=str(path), tool="run_python")


def test_only_image_and_table_are_showable(tmp_path):
    arts = [_artifact(tmp_path, "c.png", "image", PNG_1PX),
            _artifact(tmp_path, "t.csv", "table", b"a,b\n1,2\n"),
            _artifact(tmp_path, "log.txt", "text", b"hello")]
    shown = views.outputs_from_artifacts(arts)
    assert [o.kind for o in shown] == ["image", "table"]
    assert "🖼 график" in shown[0].caption and "📊 таблица" in shown[1].caption


def test_vanished_file_is_not_shown(tmp_path):
    """Заголовок «график» без картинки хуже, чем отсутствие заголовка."""
    art = _artifact(tmp_path, "gone.png", "image", PNG_1PX)
    os.remove(art.path)
    assert views.outputs_from_artifacts([art]) == []


def test_turn_outputs_shows_only_this_turn(tmp_path):
    s = _session()
    old = s.add_artifact(_artifact(tmp_path, "old.png", "image", PNG_1PX))
    fresh = s.add_artifact(_artifact(tmp_path, "new.png", "image", PNG_1PX))
    shown = views.turn_outputs(s, [fresh.id])
    assert [o.name for o in shown] == ["new.png"]
    assert old.id not in [o.name for o in shown]


def test_artifact_outputs_shows_latest_first(tmp_path):
    s = _session()
    s.add_artifact(_artifact(tmp_path, "first.png", "image", PNG_1PX))
    s.add_artifact(_artifact(tmp_path, "second.png", "image", PNG_1PX))
    shown = views.artifact_outputs(s, limit=1)
    assert [o.name for o in shown] == ["second.png"]
