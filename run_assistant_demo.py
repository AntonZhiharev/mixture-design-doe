"""run_assistant_demo.py — ПОКАЗ слоя ассистента без запуска приложения.

Демонстрационный прогон печатает ровно то, что увидит пользователь в доке
интерфейса: те же чистые хелперы (`src/assistant/views.py`), что будет рисовать
Streamlit. Скрипт растёт вместе со слоем — по одному разделу на итерацию
(ASSISTANT_SPEC).

Запуск:
    .venv\\Scripts\\python.exe run_assistant_demo.py

Артефакты кладутся в `project_campaigns/_assistant_demo/` (каталог в
.gitignore) — так же, как у настоящего проекта кампании.
"""
from __future__ import annotations

import os
import shutil
import sys

import pandas as pd

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Вывод — UTF-8 даже при перенаправлении в файл/пайп: под Windows консоль
# отдаёт cp1252, и рамки таблиц с кириллицей роняли бы демо на print().
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover — экзотические среды
        pass


from src.assistant import files, llm, store, views  # noqa: E402

from src.assistant.session import (Artifact, StagedPatch,  # noqa: E402
                                    ToolCall, new_session)


CAMPAIGN_ROOT = os.path.join(_REPO, "project_campaigns")
DEMO_PROJECT = "_assistant_demo"

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)


def head(title: str) -> None:
    print("\n" + "═" * 100)
    print(f"  {title}")
    print("═" * 100)


def show(df: pd.DataFrame, empty: str = "(пусто)") -> None:
    if df.empty:
        print(f"   {empty}")
    else:
        print(df.to_string(index=False))


# ----------------------------------------------------------------------
# iter58 — сессия, привязанная к проекту
# ----------------------------------------------------------------------
def demo_iter58() -> None:
    head("iter58 · СЕССИЯ АССИСТЕНТА, ПРИВЯЗАННАЯ К ПРОЕКТУ")

    demo_dir = os.path.join(CAMPAIGN_ROOT, DEMO_PROJECT)
    if os.path.isdir(demo_dir):
        shutil.rmtree(demo_dir)

    # --- разбор куска спеки ПВХ: технолог ↔ архитектор -----------------
    s = new_session(DEMO_PROJECT, model="anthropic/claude-sonnet-4.5",
                    web_enabled=True)
    s.add_message("user", "Три смазки (DL_60, AKLUB_K_435, OPE) хочу в группу "
                          "LUB, закрытие повесим на OPE.")
    s.add_tool_call(ToolCall(tool="get_spec", args={}, ok=True, duration_s=0.02,
                             summary="pvc_edge_v1: q=19, dim_z=16, схема v2"))
    s.add_tool_call(ToolCall(
        tool="validate_spec",
        args={"patch": {"node": "LUB.OPE", "role": "SHARE_CLOSURE"}},
        ok=False, duration_s=0.11,
        error="k=3 ⇒ SHARE_CLOSURE запрещён: все члены группы — SHARE_SIMPLEX"))
    s.add_message(
        "assistant",
        "Нельзя: при k≥3 замыкание — внутреннее свойство сэмплера на симплексе, "
        "а не роль узла. Все три смазки должны быть SHARE_SIMPLEX; группа даёт "
        "k−1 = 2 свободные координаты. Проверено validate_spec.",
        model="anthropic/claude-sonnet-4.5",
        usage={"prompt_tokens": 4120, "completion_tokens": 180,
               "total_tokens": 4300})
    s.add_message("user", "Хорошо. А верх со-стабилизатора SBM_55 поставь 0,5 — "
                          "у нас оптимум был там.")
    s.add_message(
        "assistant",
        "Не советую упирать диапазон в оптимум: вторая ветвь отказа станет "
        "ненаблюдаемой. Предлагаю 0,15–0,70 (охват ~[0.3X, 1.4X]).",
        model="anthropic/claude-sonnet-4.5",
        usage={"prompt_tokens": 4600, "completion_tokens": 90,
               "total_tokens": 4690})

    # --- вложение: реальный файл (iter59) -------------------------------
    files.attach_file(
        s, CAMPAIGN_ROOT, "TDS_Chalk_1T.txt",
        ("ПАСПОРТ: Мел природный молотый, марка 1Т\n"
         "Влажность, % ................ не более 0,3\n"
         "Белизна ISO ................. 87\n"
         "Стеаратное покрытие ......... нет\n"
         "d50, мкм .................... НЕ УКАЗАН\n"
         "Topcut, мкм ................. НЕ УКАЗАН\n").encode("utf-8"),
        note="d50/topcut отсутствуют → P_max остаётся НЕАКТИВНЫМ")


    # --- предложенные патчи (стейдж, НЕ применены) ----------------------
    s.stage_patch(StagedPatch(
        node="LUB.OPE", field_name="role", from_value="SHARE_CLOSURE",
        to_value="SHARE_SIMPLEX", bound_type="PHYSICAL", level="L3",
        source="validate_spec", confidence="high", affects_hash=True,
        rationale="k=3: closure запрещён инвариантом схемы v2 (iter46)."))
    s.stage_patch(StagedPatch(
        node="SBM_55", field_name="range", from_value=[0.5, 0.5],
        to_value=[0.15, 0.70], bound_type="CONVENTIONAL", level="L1",
        source="практика цеха (оптимум ≈0,5)", confidence="med",
        affects_hash=True,
        rationale="Диапазон охватывает оптимум, обе ветви отказа наблюдаемы."))
    s.stage_patch(StagedPatch(
        node="Chalk_1T", field_name="max_phr", from_value=None, to_value=None,
        bound_type="", level="L2", source="OPEN_QUESTION", confidence="low",
        affects_hash=False,
        rationale="P_max по крупной фракции остаётся НЕАКТИВНЫМ: d50/topcut "
                  "мела в паспорте отсутствуют — запрос поставщику."))

    # --- артефакт песочницы (полноценно — на iter62) --------------------
    s.add_artifact(Artifact(
        name="hi_phi_of_T.png", kind="image",
        path="artifacts/hi_phi_of_T.png", tool="plot",
        caption="hi_φ(T)=min(0.70, 8/T, 1−3/T) — немонотонна: 0.40@T=5, "
                "полка 0.70, 0.533@T=15"))

    # --- журналы: аудит вызовов и РЕШЕНИЯ КОМПАНИИ ----------------------
    for c in s.tool_calls:
        store.append_log(CAMPAIGN_ROOT, DEMO_PROJECT, "tool_calls", c.to_state())
    store.append_log(CAMPAIGN_ROOT, DEMO_PROJECT, "decisions", {
        "ts": "2026-08-08T15:10:00+00:00",
        "title": "LUB (k=3) — все члены SHARE_SIMPLEX",
        "nodes": ["DL_60", "AKLUB_K_435", "OPE"], "author": "технолог",
        "spec_hash": "c63b7e1696e1c449",
        "rationale": "closure при k≥3 запрещён; 22→16 координат, rank(Z)=dim"})
    store.append_log(CAMPAIGN_ROOT, DEMO_PROJECT, "decisions", {
        "ts": "2026-08-08T15:22:00+00:00",
        "title": "FILLER.total расширен 5–25 → 2–100 phr",
        "nodes": ["FILLER.total"], "author": "технолог",
        "spec_hash": "c63b7e1696e1c449",
        "rationale": "L1-факт цеха: для белых компаундов мел доходит до 100 phr; "
                     "справочные 5–25 неприменимы"})
    store.append_log(CAMPAIGN_ROOT, DEMO_PROJECT, "local_facts", {
        "ts": "2026-08-08T15:22:00+00:00", "scope": "cost",
        "statement": "Плотность компаунда ИЗМЕРЯЕТСЯ, не считается из компонентов",
        "author": "технолог"})

    print(f"\n📌 {views.session_caption(s)}")
    print(f"   {views.context_caption(s, max_tokens=24000)}")

    print("\n💬 Лента диалога (док справа рисует её же):")
    show(views.messages_dataframe(s))

    print("\n📎 Приложенные файлы сессии:")
    show(views.attachments_dataframe(s))

    print("\n🧩 Патчи спеки — СТЕЙДЖ, применяет только человек кнопкой:")
    show(views.staged_patches_dataframe(s, only_staged=True))

    print("\n🖼 Артефакты песочницы:")
    show(views.artifacts_dataframe(s))

    print("\n🔧 Аудит вызовов инструментов (assistant/tool_calls.jsonl):")
    show(views.tool_calls_dataframe(
        store.read_log(CAMPAIGN_ROOT, DEMO_PROJECT, "tool_calls")))

    print("\n📚 Решения компании (assistant/decision_log.jsonl):")
    show(views.decisions_dataframe(
        store.read_log(CAMPAIGN_ROOT, DEMO_PROJECT, "decisions")))

    # --- ПЕРСИСТЕНТНОСТЬ: закрыли приложение → открыли проект -----------
    path = store.save_session(s, CAMPAIGN_ROOT)
    reloaded = store.load_session(CAMPAIGN_ROOT, DEMO_PROJECT)
    print(f"\n💾 Сессия сохранена: {os.path.relpath(path, _REPO)}")
    print(f"♻️  Проект открыт заново → {views.session_caption(reloaded)}")
    assert reloaded.to_state() == s.to_state(), "round-trip сессии нарушен"

    # --- усечение контекста НЕ теряет историю ---------------------------
    for i in range(60):
        reloaded.add_message("user", f"уточнение №{i}: " + "детали " * 60)
    print("\n✂️  Длинный разбор (66 сообщений), бюджет контекста 2000 токенов:")
    print(f"   {views.context_caption(reloaded, max_tokens=2000)}")
    ctx = reloaded.context_messages(max_tokens=2000)
    print(f"   первая строка контекста → {ctx[0]['content'][:96]}…")

    print(f"\n📁 Всё лежит рядом с проектом: "
          f"{os.path.relpath(store.assistant_dir(CAMPAIGN_ROOT, DEMO_PROJECT), _REPO)}")
    for entry in sorted(os.listdir(store.assistant_dir(CAMPAIGN_ROOT, DEMO_PROJECT))):
        print(f"      • {entry}")
    print("   (удаление проекта уносит переписку, файлы и решения — "
          "отдельной синхронизации нет)")


# ----------------------------------------------------------------------
# iter59 — вложения: паспорта, выгрузки, протоколы
# ----------------------------------------------------------------------
def _demo_xlsx() -> bytes:
    """Мини-выгрузка лаборатории (как реальный файл из цеха)."""
    import io

    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Опыты"
    ws.append(["№ опыта", "Gloss_60", "Adhesion_N", "dE"])
    for row in ([1, 8.4, 3.2, 0.9], [2, 11.7, 2.9, 0.6], [3, 6.1, 3.8, 1.4]):
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _demo_pdf(text: str) -> bytes:
    """PDF с текстовым слоем — самый частый вид паспорта сырья (TDS)."""
    import io

    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 780
    for line in text.splitlines():
        c.drawString(60, y, line)
        y -= 18
    c.showPage()
    c.save()
    return buf.getvalue()


def demo_iter59() -> None:
    head("iter59 · ФАЙЛЫ В СЕССИИ: паспорта, выгрузки, протоколы")

    s = store.load_session(CAMPAIGN_ROOT, DEMO_PROJECT)

    print("\n➕ Прикладываем документы (файлы сохраняются В ПРОЕКТ):")
    s.add_message("user", "Прикладываю паспорт УФ-абсорбера и выгрузку "
                          "лаборатории — посмотри, что можно закрыть по "
                          "открытым вопросам.")

    pdf_bytes = _demo_pdf("TECHNICAL DATA SHEET: UV-CSFCP\n"
                          "Melting point, C .......... 128 - 132\n"
                          "Solubility in DINP ........ 3 % w/w max\n"
                          "CAS ....................... 3896-11-5\n"
                          "Bulk density, g/cm3 ....... 0.45")
    pdf = files.attach_file(s, CAMPAIGN_ROOT, "TDS_UV_CSFCP.pdf", pdf_bytes)
    print(f"   • {pdf.name}: извлечено {pdf.n_chars} симв., тип {pdf.mime}")

    xlsx = files.attach_file(s, CAMPAIGN_ROOT, "lab_runs.xlsx", _demo_xlsx())
    print(f"   • {xlsx.name}: извлечено {xlsx.n_chars} симв. "
          f"(листы и ячейки — текстом)")

    # дедуп: ТОТ ЖЕ файл прислали второй раз под другим именем
    n_before = len(s.attachments)
    same = files.attach_file(s, CAMPAIGN_ROOT, "TDS_UV_CSFCP (копия).pdf",
                             pdf_bytes)
    print(f"   • «копия» того же PDF → дедуп по sha256: записей было "
          f"{n_before}, стало {len(s.attachments)}; вернулось прежнее "
          f"вложение «{same.name}»")


    # A0.6: отказы объясняют себя
    print("\n⛔ Отказы объясняют причину (A0.6):")
    for bad_name, payload in (("photo_of_screen.png", b"\x89PNG\r\n"),
                              ("empty.txt", b""),
                              ("broken.xlsx", b"not a workbook")):
        try:
            files.attach_file(s, CAMPAIGN_ROOT, bad_name, payload)
        except files.AttachmentError as exc:
            print(f"   • {bad_name}: {str(exc)[:96]}…")

    print("\n📎 Файлы сессии:")
    show(views.attachments_dataframe(s))

    print("\n🔍 Что из паспорта видит ассистент (фрагмент по запросу):")
    got = files.attachment_text(s, CAMPAIGN_ROOT, "TDS_UV_CSFCP.pdf",
                                start=0, length=220)
    for line in got["text"].strip().splitlines()[:6]:
        print(f"      {line}")
    print(f"      … всего {got['total_chars']} симв., есть продолжение: "
          f"{got['has_more']}")

    s.add_message(
        "assistant",
        "Из паспорта UV_CSFCP: т.пл. 128–132 °C — в dry-blend при T_mix "
        "100–120 °C НЕ плавится ⇒ узел идёт через ПРЕМИКС (CAMPAIGN_SPEC §5). "
        "Растворимость в DINP 3 % w/w подтверждает трапецию "
        "hi_UV = min(0,30; 0,03·(p_DINP + 2,50)) — это ABSOLUTE_CAPPED по фазе, "
        "а не RATIO_TO. d50 мела в приложенных документах по-прежнему НЕТ — "
        "P_max остаётся неактивным (OPEN_QUESTION).",
        model="anthropic/claude-sonnet-4.5",
        usage={"prompt_tokens": 7300, "completion_tokens": 210,
               "total_tokens": 7510})

    print("\n💬 Ответ по документам (без выдумывания недостающего):")
    show(views.messages_dataframe(s).tail(2))

    print(f"\n📌 {views.session_caption(s)}")
    print(f"   дайджест в контекст: "
          f"{[(d['name'], d['n_chars'], d['clipped']) for d in views.attachment_digest(s)]}")

    store.save_session(s, CAMPAIGN_ROOT)
    fdir = store.files_dir(CAMPAIGN_ROOT, DEMO_PROJECT)
    print(f"\n📁 Файлы на диске проекта ({os.path.relpath(fdir, _REPO)}):")
    for entry in sorted(os.listdir(fdir)):
        print(f"      • {entry}")
    print(f"   сироты (в сессии больше не числятся): "
          f"{files.orphan_files(s, CAMPAIGN_ROOT) or 'нет'}")


# ----------------------------------------------------------------------
# iter60 — ход ассистента: модель ↔ инструменты (+ интернет, + прогресс)
# ----------------------------------------------------------------------
def _scripted_transport(script):
    """Заглушка сети: демо показывает ПОВЕДЕНИЕ цикла, а не ответ конкретной
    модели (для живого ответа нужен ключ OpenRouter — см. панель ассистента)."""
    queue = list(script)

    def _transport(payload, *, key="", timeout=0):
        if not queue:
            raise AssertionError("сценарий демо исчерпан")
        return queue.pop(0)

    return _transport


def demo_iter60() -> None:
    head("iter60 · ХОД АССИСТЕНТА: модель ↔ инструменты (tool-loop)")

    import json as _json

    s = store.load_session(CAMPAIGN_ROOT, DEMO_PROJECT)
    question = ("Привяжи УФ-абсорбер к пигменту: их дозируют вместе, "
                "поставь RATIO_TO.")
    s.add_message("user", question)

    def _fn(name, args):
        return {"id": f"c_{name}", "type": "function",
                "function": {"name": name, "arguments": _json.dumps(args)}}

    def _answer(content, tool_calls=None, usage=None):
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        body = {"choices": [{"message": msg}]}
        if usage:
            body["usage"] = usage
        return body

    # Сценарий: модель сверяется со спекой → считает корреляции → отвечает
    script = [
        _answer("", [_fn("get_spec", {})], {"total_tokens": 5200}),
        _answer("", [_fn("simulate_bounds",
                         {"patch": {"node": "UV_CSFCP", "role": "RATIO_TO",
                                    "reference": "TiO2_BLR895"}})],
                {"total_tokens": 1800}),
        _answer(
            "Отказ: TiO2 не задаёт эффективность УФ-абсорбера, он её ЗАМЕЩАЕТ "
            "(антагонизм по светостойкости) — RATIO_TO вшил бы неверный "
            "монотонный prior, который модель не сможет опровергнуть. "
            "simulate_bounds: клин даёт corr(p_UV, p_TiO2) = 0.91, корректная "
            "трапеция по пластификаторной фазе — 0.12. Оставляем "
            "ABSOLUTE_CAPPED (cap_to = фаза DINP) + interacts_with = "
            "TiO2_BLR895. Совместное дозирование — вопрос ПРЕМИКСА, а не "
            "геометрии спеки.", None, {"total_tokens": 900}),
    ]

    # Инструменты исполняются локально (полноценный реестр — iter61)
    def dispatch(name, args):
        if name == "get_spec":
            return {"spec_version": 2, "q": 19, "dim_z": 16,
                    "spec_hash": "c63b7e1696e1c449",
                    "group_order": ["FILLER.total", "SOFT.total", "ACR.total",
                                    "LUB.total"],
                    "UV_CSFCP": {"role": "ABSOLUTE_CAPPED", "scale": "log",
                                 "range": [0.05, 0.30],
                                 "cap_to": ["DINP", "ESO"], "cap_ratio": 0.03}}
        if name == "simulate_bounds":
            return {"proposed": {"role": "RATIO_TO", "corr_with_reference": 0.91},
                    "current": {"role": "ABSOLUTE_CAPPED",
                                "corr_with_reference": 0.12},
                    "sigma_phr": [115.2, 249.8], "warning":
                        "клин делает узел почти линейной функцией референса"}
        raise ValueError(f"инструмент '{name}' не зарегистрирован")

    print(f"\n👤 Вопрос: {question}")
    print("\n⏳ Прогресс хода (это же увидит пользователь под чатом):")
    events = []

    def on_event(ev):
        events.append(ev)
        print(f"   {llm.progress_caption(ev)}")

    res = llm.run_tool_loop(s.context_messages(max_tokens=8000),
                            dispatch=dispatch, model=s.model,
                            web=s.web_enabled, transport=_scripted_transport(script),
                            on_event=on_event)

    print(f"\n🤖 Ответ (модель `{res.model}`, шагов {res.iterations}, "
          f"вызовов {res.n_tool_calls}, итог: {res.stopped_reason}):")
    for line in res.text.splitlines():
        print(f"   {line}")

    # ход целиком ложится в память проекта
    for m in res.new_messages:
        s.add_message(m["role"], m.get("content", ""),
                      tool_calls=m.get("tool_calls", []),
                      tool_call_id=m.get("tool_call_id", ""),
                      name=m.get("name", ""),
                      model=res.model if m["role"] == "assistant" else "",
                      web=res.web if m["role"] == "assistant" else False)
    s.add_usage(res.usage)
    for c in res.calls:
        rec = ToolCall(tool=c["tool"], args=c["args"], ok=c["ok"],
                       error=c["error"], duration_s=c["duration_s"],
                       summary=c["summary"])
        s.add_tool_call(rec)
        store.append_log(CAMPAIGN_ROOT, DEMO_PROJECT, "tool_calls", rec.to_state())

    print("\n🔧 Что реально вызывалось (аудит хода):")
    show(views.tool_calls_dataframe(res.calls))

    print("\n💬 Лента после хода (роль «инструмент» — часть переписки):")
    show(views.messages_dataframe(s).tail(4))

    print(f"\n📌 {views.session_caption(s)}")
    print("   🌐 интернет включён ⇒ модель ушла с суффиксом ':online' "
          f"(`{res.model}`) — источник ответа виден в ленте")

    store.save_session(s, CAMPAIGN_ROOT)
    print("\n⛔ Отказ инструмента возвращается МОДЕЛИ, а не пользователю:")
    bad = llm.run_tool_loop(
        [{"role": "user", "content": "?"}], dispatch=dispatch,
        transport=_scripted_transport([
            _answer("", [_fn("get_local_facts", {"scope": "cost"})]),
            _answer("Инструмент недоступен — отвечаю по контексту проекта.")]))
    print(f"   вызов: {bad.calls[0]['tool']} · ok={bad.calls[0]['ok']} · "
          f"{bad.calls[0]['error']}")
    print(f"   ответ модели: {bad.text}")


# ----------------------------------------------------------------------
# iter61 — read-only инструменты: ответ приходит ИЗ ЯДРА, а не по памяти
# ----------------------------------------------------------------------
#: Референсная v2-спека показа — та же геометрия, что в golden-тестах
#: (iter45/49/50): SOFT-группа с техлимитами, лог-оси, cap-трапеция УФ.
DEMO_NODES = [
    {"name": "RESIN", "role": "FIXED", "value": 100.0},
    {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
    {"name": "ESO", "role": "FIXED", "value": 2.5},
    {"name": "SOFT", "role": "GROUP_TOTAL", "range": [5.0, 15.0],
     "members": ["PBNK", "CPE"]},
    {"name": "PBNK", "role": "SHARE_FREE", "group": "SOFT",
     "share_range": [0.0, 0.70], "max_phr": 8.0},
    {"name": "CPE", "role": "SHARE_CLOSURE", "group": "SOFT", "min_phr": 3.0},
    {"name": "TiO2", "role": "ABSOLUTE", "range": [0.3, 8.0], "scale": "log"},
    {"name": "UV", "role": "ABSOLUTE_CAPPED", "range": [0.05, 0.30],
     "scale": "log", "cap_to": ["DINP", "ESO"], "cap_ratio": 0.03},
]

#: «Привяжи УФ к пластификатору жёстко» — КЛИН вместо трапеции по фазе.
DEMO_WEDGE = {"node": "UV",
              "set": {"role": "RATIO_TO", "reference": "DINP",
                      "range": [0.0125, 0.0214]},
              "unset": ["cap_to", "cap_ratio", "scale"]}


def demo_iter61() -> None:
    head("iter61 · READ-ONLY ИНСТРУМЕНТЫ: числа из ядра, а не из памяти")

    import json as _json

    from src.assistant.tools import ToolContext, ToolError, dispatch, tool_specs
    from src.assistant.tools.registry import READONLY, dispatcher, is_long_running
    from src.design.phr_sampler import PhrSpec

    s = store.load_session(CAMPAIGN_ROOT, DEMO_PROJECT)
    spec = PhrSpec.from_dicts(DEMO_NODES)
    ctx = ToolContext(spec=spec, session=s, root=CAMPAIGN_ROOT,
                      project=DEMO_PROJECT)

    print("\n🧰 Инструменты, которые видит модель (класс readonly):")
    rows = [{"инструмент": f["function"]["name"],
             "долгий": "да" if is_long_running(f["function"]["name"]) else "",
             "аргументы": ", ".join(f["function"]["parameters"]["properties"])
                          or "—"}
            for f in tool_specs([READONLY])]
    show(pd.DataFrame(rows))

    def h(value: str) -> str:
        """Короткий вид отпечатка: в тексте важен ФАКТ совпадения/сдвига."""
        return f"{str(value)[:16]}…"

    # --- 1. get_spec: снимок геометрии --------------------------------
    snap = dispatch(ctx, "get_spec", {"include_nodes": False})
    print(f"\n📐 get_spec → spec_hash={h(snap['spec_hash'])} · "
          f"q={snap['q_components']} · dim_z={snap['dim_z']} · "
          f"лог-оси {snap['log_axes']} · Σphr(статически) "
          f"{snap['sigma_phr_static']}")

    # --- 2. explain_node: НЕМОНОТОННОСТЬ hi_φ(T) ----------------------
    node = dispatch(ctx, "explain_node",
                    {"name": "PBNK", "totals": [5.0, 7.5, 10.5, 12.5, 15.0]})
    print("\n🔎 explain_node('PBNK') — «почему верх доли не 0,70, как я ввёл»:")
    show(pd.DataFrame(node["effective_shares"]).round(4))
    print("   hi_φ(T) = min(0.70 · 8/T · 1−3/T): 0.40 @T=5 → полка 0.70 → "
          "0.5333 @T=15 — по двум точкам вывод сделать НЕЛЬЗЯ.")
    print(f"   L1-факты цеха по узлу: "
          f"{[f['statement'] for f in node.get('local_facts', [])] or 'нет'}")

    cap = dispatch(ctx, "explain_node", {"name": "UV"})["cap"]
    print(f"\n🔎 explain_node('UV') → cap {cap['cap_ratio']}·Σ{cap['cap_to']} "
          f"в точке: {cap['note'].splitlines()[0]}")

    # --- 3. validate_spec: dry-run патча ------------------------------
    print("\n🧪 validate_spec — сухой прогон, проект не меняется:")
    ok = dispatch(ctx, "validate_spec",
                  {"patch": {"node": "DINP", "field": "range",
                             "value": [4.0, 20.0]}})
    print(f"   • DINP 4–14 → 4–20: ok={ok['ok']} · "
          f"отпечаток едет: {ok['affects_hash']} "
          f"({h(ok['spec_hash_before'])} → {h(ok['spec_hash_after'])})")
    for d in ok["changed_intervals"]:
        print(f"       {d['node']}: {d['before']} → {d['after']}")
    bad = dispatch(ctx, "validate_spec",
                   {"patch": {"node": "CPE", "field": "share_range",
                              "value": [0.1, 0.9]}})
    print(f"   • CPE (SHARE_CLOSURE) задать share_range: ok={bad['ok']} · "
          f"{bad['error']}")
    print(f"     {bad['hint']}")
    print(f"   спека проекта после dry-run: {h(spec.spec_hash())} (не тронута)")

    # --- 4. simulate_bounds: КЛИН против ТРАПЕЦИИ ---------------------
    sim = dispatch(ctx, "simulate_bounds",
                   {"patch": DEMO_WEDGE, "n": 400, "seed": 0,
                    "pair": ["UV", "DINP"]})
    print("\n📊 simulate_bounds — тот самый аргумент против «привяжи УФ к "
          "пластификатору»:")
    show(pd.DataFrame([
        {"вариант": "сейчас · ABSOLUTE_CAPPED (трапеция по фазе)",
         "corr(UV,DINP)": sim["current"]["pair_corr"],
         "Σphr": [round(v, 1) for v in sim["current"]["sigma_phr"]]},
        {"вариант": "патч · RATIO_TO (клин)",
         "corr(UV,DINP)": sim["proposed"]["pair_corr"],
         "Σphr": [round(v, 1) for v in sim["proposed"]["sigma_phr"]]},
    ]))
    print(f"   сдвиг корреляции: +{sim['pair_corr_shift']} ⇒ клин вшивает "
          f"монотонный prior, который данные уже НЕ ОПРОВЕРГНУТ.")

    # --- 5. point_report: разбор конкретного рецепта ------------------
    recipe = [100.0, 6.0, 2.5, 8.0, 7.0, 1.0, 0.10]   # T_soft = 15, PBNK = 8
    rep = dispatch(ctx, "point_report",
                   {"recipe_phr": recipe, "delta_phr": 0.02})
    pbnk = rep["effective_bounds"]["PBNK"]
    print(f"\n🧾 point_report(рецепт с T_soft=15, PBNK=8 phr): ok={rep['ok']}")
    print(f"   PBNK: доля {pbnk['coord']:.4f} = {pbnk['phr']:.2f} phr, верх "
          f"держит «{pbnk['active_hi']}» (складской лимит 8 phr, не share_range)")
    print(f"   премикс нужен для: "
          f"{[k for k, v in rep['premix'].items() if v] or 'нет'} "
          f"(шаг весов δ=0.02 phr)")
    over = dispatch(ctx, "encode_recipe",
                    {"recipe_phr": [100.0, 25.0, 2.5, 5.0, 5.0, 1.0, 0.10]})
    print(f"   серийный рецепт с DINP=25: представим={over['representable']} "
          f"⇒ {over['hint'].splitlines()[0]}")

    # --- 6. A0.6: отказы объясняют себя -------------------------------
    print("\n⛔ Отказы объясняют причину (уходят МОДЕЛИ как результат):")
    for name, args in (("explain_node", {"name": "Chalk_1T"}),
                       ("simulate_bounds", {"n": 50, "pair": ["UV", "Chalk"]}),
                       ("get_runs", {})):
        try:
            dispatch(ctx, name, args)
        except ToolError as exc:
            print(f"   • {name}: {str(exc)[:110]}…")

    # --- 7. Стыковка с ходом модели (iter60) --------------------------
    print("\n🔁 Тот же ход, что в iter60, но инструменты НАСТОЯЩИЕ:")
    audit = []
    call = dispatcher(ctx, on_call=audit.append)
    script = [
        {"choices": [{"message": {"role": "assistant", "content": "",
                                  "tool_calls": [{"id": "c1", "function": {
                                      "name": "explain_node",
                                      "arguments": _json.dumps(
                                          {"name": "PBNK",
                                           "totals": [10.5, 15.0]})}}]}}]},
        {"choices": [{"message": {"role": "assistant", "content": (
            "Верх доли PBNK не постоянен: при T=10,5 это 0,70, при T=15 — "
            "0,533 (давит складской лимит 8 phr). Диапазон в спеке НЕ "
            "занижен — сужение делает conditional narrowing.")}}]},
    ]
    res = llm.run_tool_loop(
        [{"role": "user", "content": "Почему верх PBNK не 0,70?"}],
        dispatch=call, tools=tool_specs([READONLY]),
        transport=_scripted_transport(script))
    tool_msg = [m for m in res.new_messages if m["role"] == "tool"][0]
    print(f"   инструмент вернул модели {len(tool_msg['content'])} симв. "
          f"настоящих чисел ядра (spec_hash={h(spec.spec_hash())})")
    for line in res.text.splitlines():
        print(f"   🤖 {line}")

    print("\n🔧 Аудит вызовов (пишется в assistant/tool_calls.jsonl):")
    for rec in audit:
        store.append_log(CAMPAIGN_ROOT, DEMO_PROJECT, "tool_calls",
                         {"tool": rec["tool"], "args": rec["args"],
                          "ok": rec["ok"], "error": rec["error"],
                          "duration_s": rec["duration_s"],
                          "summary": rec.get("summary", "")})
    show(views.tool_calls_dataframe(audit))


# ----------------------------------------------------------------------
# iter62 — песочница: проверка исполнением, а не убеждением
# ----------------------------------------------------------------------
#: Тесты-однодневки для показа `run_pytest` с прогрессом: настоящий прогон
#: профильного файла кампании занял бы минуту, а показать надо механику.
DEMO_TESTS = '''\
def test_share_hi_at_T15():
    assert round(min(0.70, 8 / 15, 1 - 3 / 15), 4) == 0.5333


def test_share_hi_shelf():
    assert min(0.70, 8 / 10.5, 1 - 3 / 10.5) == 0.70


def test_wrong_expectation():
    assert min(0.70, 8 / 15, 1 - 3 / 15) == 0.70, "верх доли НЕ постоянен"
'''


def demo_iter62() -> None:
    head("iter62 · ПЕСОЧНИЦА: тайм-аут, отказ сети, отказ записи, run_pytest")

    import tempfile

    from src.assistant.sandbox import (SandboxPolicy, SubprocessSandbox,
                                       progress_caption)
    from src.assistant.tools import ToolContext, dispatch
    from src.assistant.tools.registry import SANDBOX

    scratch = tempfile.mkdtemp(prefix="doe_demo_sandbox_")
    sb = SubprocessSandbox(SandboxPolicy(repo_root=_REPO,
                                         workdir=os.path.join(scratch, "work"),
                                         timeout_s=60.0))
    try:
        info = sb.describe()
        print(f"\n🧱 Бэкенд `{info['backend']}` · репозиторий: "
              f"{info['repo_access']} · сеть: {info['network']} · тайм-аут "
              f"{info['timeout_s']:.0f} с")
        print(f"   писать можно только сюда: {info['write_roots'][0]}")
        print(f"   защищено отдельно: {info['protected']}")

        # --- 1. считает ТО, для чего нет готового инструмента -----------
        res = sb.run_python(
            "T = 15.0\n"
            "hi = min(0.70, 8 / T, 1 - 3 / T)\n"
            "print(f'hi_phi(T=15) = {hi:.4f}')")
        print(f"\n🧮 run_python — счёт вместо рассуждения: {res.caption()}")
        for line in res.stdout.splitlines():
            print(f"   {line}")

        # --- 2. отказ сети ---------------------------------------------
        net = sb.run_python("import socket\nsocket.socket().connect(('1.1.1.1', 80))")
        print(f"\n🌐 Попытка выйти в сеть: {net.caption()}")
        print(f"   {net.note}")

        # --- 3. отказ записи -------------------------------------------
        target = os.path.join(_REPO, "tests", "unit",
                              "test_iteration45_phr_spec.py").replace("\\", "\\\\")
        wr = sb.run_python(f"open('{target}', 'a').write('# подправлю тест')")
        print(f"\n✍️  Попытка подправить тест: {wr.caption()}")
        print(f"   {wr.note}")

        # --- 4. тайм-аут ------------------------------------------------
        slow = sb.run_python("import time\nprint('считаю…', flush=True)\n"
                             "time.sleep(30)", timeout_s=2.0)
        print(f"\n⏱ Зависший расчёт: {slow.caption()} "
              f"(частичный вывод: {slow.stdout.strip()!r})")

        # --- 5. run_pytest с прогрессом ---------------------------------
        target_tests = os.path.join(scratch, "test_demo_shares.py")
        with open(target_tests, "w", encoding="utf-8") as fh:
            fh.write(DEMO_TESTS)

        print("\n🧪 run_pytest — так проверяется «патч ничего не ломает» "
              "(строки ниже видит пользователь в доке):")
        rep = sb.run_pytest([target_tests], timeout_s=180,
                            on_progress=lambda ev: print(
                                f"   {progress_caption(ev)}"))
        print(f"\n   итог: {rep.caption()}")
        print(f"   упало: {[os.path.basename(f) for f in rep.failures]}")

        # --- 6. то же самое, но как ИНСТРУМЕНТ модели -------------------
        s = store.load_session(CAMPAIGN_ROOT, DEMO_PROJECT)
        ctx = ToolContext(session=s, root=CAMPAIGN_ROOT, project=DEMO_PROJECT,
                          extra={"sandbox": sb})
        out = dispatch(ctx, "run_pytest", {"targets": [target_tests]},
                       allowed_kinds=[SANDBOX])
        print(f"\n🔧 Инструмент run_pytest вернул модели: ok={out['ok']} · "
              f"{out['passed']} прошло · {out['failed']} упало · "
              f"артефакт сохранён в кампанию")
        print("\n🗂 Артефакты сессии (выхлоп песочницы живёт в проекте):")
        show(views.artifacts_dataframe(s))
        store.save_session(s, CAMPAIGN_ROOT)
    finally:
        sb.close()
        shutil.rmtree(scratch, ignore_errors=True)


# ----------------------------------------------------------------------
# iter63 — предлагает МОДЕЛЬ, применяет ЧЕЛОВЕК
# ----------------------------------------------------------------------
class _DemoRunner:
    """Проект кампании в объёме, который нужен гейтам применения.

    Настоящий раннер сюда тащить незачем: показываем, что гейт смотрит на
    ОБЩУЮ БАЗУ ТОЧЕК и на preflight, а не на слова ассистента.
    """

    def __init__(self, spec, X):
        self.phr_spec = spec
        self.X = X

    def set_phr_spec(self, spec):
        self.phr_spec = spec

    def preflight(self, X):                     # гейты кампании не ухудшаются
        class _R:
            passed = True
            failures: list = []
        return _R()


def demo_iter63() -> None:
    head("iter63 · WRITE-ИНСТРУМЕНТЫ: предлагает модель, применяет человек")

    import json as _json

    import numpy as np

    from src.assistant.consent import ConsentRegistry
    from src.assistant.tools import (AGENT_KINDS, PROPOSE, WRITE, ToolContext,
                                     ToolError, dispatch, tool_names,
                                     tool_specs)
    from src.assistant.tools.registry import dispatcher
    from src.assistant.tools.write import issue_apply_token, issue_reject_token
    from src.design.phr_sampler import PhrSpec

    s = store.load_session(CAMPAIGN_ROOT, DEMO_PROJECT)
    spec = PhrSpec.from_dicts(DEMO_NODES)
    # одна УЖЕ ИЗМЕРЕННАЯ точка проекта: DINP = 13 phr (верх прежней области)
    measured = np.atleast_2d(spec.to_fractions(
        np.array([100.0, 13.0, 2.5, 7.0, 8.0, 1.0, 0.10])))
    runner = _DemoRunner(spec, measured)
    consent = ConsentRegistry()
    ctx = ToolContext(spec=spec, runner=runner, session=s, root=CAMPAIGN_ROOT,
                      project=DEMO_PROJECT, extra={"consent": consent})

    print(f"\n🧰 Модели выдаются классы {list(AGENT_KINDS)}:")
    print(f"   предлагать может: {tool_names([PROPOSE])}")
    print(f"   применять НЕ может (класс write): {tool_names([WRITE])}")

    # --- 1. Ход модели: числа из ядра → предложение патча ---------------
    audit = []
    call = dispatcher(ctx, allowed_kinds=AGENT_KINDS, on_call=audit.append)
    question = ("В цехе для белых компаундов льём DINP до 20 phr, а в спеке "
                "верх 14 — расширь.")
    s.add_message("user", question)
    print(f"\n👤 {question}")

    def _fn(name, args):
        return {"id": f"c_{name}", "type": "function",
                "function": {"name": name, "arguments": _json.dumps(args)}}

    def _answer(content, tool_calls=None):
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return {"choices": [{"message": msg}]}

    script = [
        _answer("", [_fn("explain_node", {"name": "DINP"})]),
        _answer("", [_fn("propose_patch", {
            "patch": {"node": "DINP", "field": "range", "value": [4.0, 20.0]},
            "rationale": "L1-факт цеха: для белых компаундов DINP доходит до "
                         "20 phr; прежний верх 14 — договорённость, а не "
                         "физический предел.",
            "bound_type": "CONVENTIONAL", "level": "L1",
            "source": "технолог (устно, 09.08.2026)", "confidence": "high"})]),
        _answer("Предложил патч: DINP.range 4–14 → 4–20 phr (CONVENTIONAL, L1). "
                "Отпечаток спеки поедет ⇒ уже собранные точки относятся к "
                "прежней геометрии. Применить может только человек — кнопка "
                "«Применить» в панели патчей."),
    ]
    res = llm.run_tool_loop([{"role": "user", "content": question}],
                            dispatch=call, tools=tool_specs(AGENT_KINDS),
                            transport=_scripted_transport(script))
    for line in res.text.splitlines():
        print(f"   🤖 {line}")

    pid = s.staged_patches()[-1].id
    print("\n🧩 Патч лёг в СТЕЙДЖ (спека проекта не тронута, "
          f"hash={spec.spec_hash()[:16]}…):")
    show(views.staged_patches_dataframe(s, only_staged=True))

    # --- 2. Модель не может применить сама -----------------------------
    print("\n⛔ Попытка применить патч САМОЙ моделью (класс write ей не выдан):")
    try:
        call("apply_patch", {"patch_id": pid, "human_token": "я-сам"})
    except ToolError as exc:
        print(f"   {str(exc)[:150]}…")

    print("\n⛔ Тот же вызов «с улицы», но без токена человека:")
    try:
        dispatch(ctx, "apply_patch", {"patch_id": pid, "human_token": ""},
                 allowed_kinds=[WRITE])
    except ToolError as exc:
        print(f"   {str(exc)[:150]}…")

    # --- 3. Гейт применения: измеренная точка выпадает из геометрии -----
    print("\n🚧 А теперь патч НАОБОРОТ — сузить верх DINP до 8 phr:")
    narrow = dispatch(ctx, "propose_patch",
                      {"patch": {"node": "DINP", "field": "range",
                                 "value": [4.0, 8.0]},
                       "rationale": "гипотеза: выше 8 phr не нужно",
                       "level": "L3", "confidence": "low"},
                      allowed_kinds=[PROPOSE])
    narrow_id = narrow["patch_ids"][0]
    try:
        dispatch(ctx, "apply_patch",
                 {"patch_id": narrow_id,
                  "human_token": issue_apply_token(ctx, narrow_id)},
                 allowed_kinds=[WRITE])
    except ToolError as exc:
        for chunk in str(exc).split(". "):
            print(f"   {chunk.strip()}")

    # --- 4. Человек отклоняет патч — отказ ТОЖЕ решение -----------------
    dispatch(ctx, "reject_patch",
             {"patch_id": narrow_id,
              "human_token": issue_reject_token(ctx, narrow_id),
              "reason": "сужение обесценивает опыт с DINP=13 phr",
              "author": "технолог"}, allowed_kinds=[WRITE])

    # --- 5. Человек нажал «Применить»: кнопка = разовый токен -----------
    token = issue_apply_token(ctx, pid, note="кнопка «Применить» в доке")
    print(f"\n🔑 Кнопка выдала разовый токен {token[:6]}… "
          f"(действие apply_patch, цель {pid}, привязан к spec_hash):")
    show(views.consents_dataframe(consent.pending()))

    out = dispatch(ctx, "apply_patch",
                   {"patch_id": pid, "human_token": token,
                    "note": "согласовано на планёрке", "author": "технолог"},
                   allowed_kinds=[WRITE])
    print(f"\n✅ {views.apply_result_caption(out)}")
    print(f"   DINP теперь: {ctx.spec.phr_intervals()['DINP']} phr; "
          f"проект получил новую спеку "
          f"({runner.phr_spec.spec_hash()[:16]}…)")
    print(f"   {out['warning']}")
    print(f"   {out['persist_hint']}")

    # --- 6. Повторное применение и токен «не от того» патча ------------
    print("\n⛔ Границы подтверждения:")
    try:
        dispatch(ctx, "apply_patch",
                 {"patch_id": pid, "human_token": issue_apply_token(ctx, pid)},
                 allowed_kinds=[WRITE])
    except ToolError as exc:
        print(f"   • тот же патч второй раз: {str(exc)[:120]}…")

    third = dispatch(ctx, "propose_patch",
                     {"patch": {"node": "TiO2", "field": "range",
                                "value": [0.3, 9.0]},
                      "rationale": "новый лот пигмента", "level": "L2"},
                     allowed_kinds=[PROPOSE])["patch_ids"][0]
    try:
        dispatch(ctx, "apply_patch",
                 {"patch_id": third, "human_token": issue_apply_token(ctx, pid)},
                 allowed_kinds=[WRITE])
    except ToolError as exc:
        print(f"   • токен ОТ ДРУГОГО патча: {str(exc)[:120]}…")

    print("\n🧩 Патчи сессии после разбора (статусы — часть памяти проекта):")
    show(views.staged_patches_dataframe(s))

    print("\n📚 Журнал решений компании (применение И отказ — обе записи):")
    show(views.decisions_dataframe(
        store.read_log(CAMPAIGN_ROOT, DEMO_PROJECT, "decisions")))

    for rec in audit:
        store.append_log(CAMPAIGN_ROOT, DEMO_PROJECT, "tool_calls",
                         {"tool": rec["tool"], "args": rec["args"],
                          "ok": rec["ok"], "error": rec["error"],
                          "duration_s": rec["duration_s"],
                          "summary": rec.get("summary", "")})
    store.save_session(s, CAMPAIGN_ROOT)
    print(f"\n📌 {views.session_caption(s)}")


# ----------------------------------------------------------------------
# iter64 — промпт архитектора и маршрутизация типовых вопросов (§8)
# ----------------------------------------------------------------------
def demo_iter64() -> None:
    head("iter64 · ПРОМПТ АРХИТЕКТОРА И МАРШРУТИЗАЦИЯ ТИПОВЫХ ВОПРОСОВ (§8)")

    import json as _json

    from src.assistant import prompts
    from src.assistant.tools import (AGENT_KINDS, ToolContext, tool_names,
                                     tool_specs)
    from src.assistant.tools.registry import dispatcher
    from src.design.phr_sampler import PhrSpec

    s = store.load_session(CAMPAIGN_ROOT, DEMO_PROJECT)
    spec = PhrSpec.from_dicts(DEMO_NODES)
    ctx = ToolContext(spec=spec, session=s, root=CAMPAIGN_ROOT,
                      project=DEMO_PROJECT)

    prompt = prompts.architect_system_prompt(
        project=DEMO_PROJECT, spec_hash=spec.spec_hash()[:16], web=True,
        n_attachments=len(s.attachments))
    print("\n📜 Системный промпт (это ИНСТРУКЦИЯ, а не «характер» помощника):")
    for line in prompts.ROLE_BLOCK.splitlines()[:4]:
        print(f"   {line}")
    print("   …")
    print("\n🎓 Иерархия знания — конфликт НЕ усредняется:")
    for line in prompts.KNOWLEDGE_BLOCK.splitlines()[:4]:
        print(f"   {line}")
    print("\n🚧 Границы (проведены кодом, в промпте — чтобы не тратить ход):")
    for line in prompts.LIMITS_BLOCK.splitlines()[1:5]:
        print(f"   {line}")

    print(f"\n🧰 Каталог инструментов собран ИЗ РЕЕСТРА "
          f"({len(tool_names(AGENT_KINDS))} шт.), класс write в него не входит "
          f"вовсе: {tool_names(['write'])}")
    print(f"   длина промпта: {len(prompt)} символов "
          f"(~{len(prompt) // 4} токенов)")

    print("\n🧭 Golden-сценарии §8 — контракт «вопрос → чем отвечаем»:")
    show(views.scenarios_dataframe())

    # --- маршрутизация: чистая функция, без сети ------------------------
    reports = []
    for sc in prompts.GOLDEN_SCENARIOS:
        r = prompts.route(sc.user)
        reports.append(prompts.check_routing(sc, r.tools))
    print("\n✅ Прогон роутера по всем восьми (сеть не нужна):")
    show(views.routing_dataframe(reports))

    print("\n🔎 Пограничные формулировки — порядок правил ЗНАЧИМ:")
    for q in ("Сузь DINP до 8 phr",
              "Что изменится, если сузить DINP до 8 phr?",
              "Расширь DINP до 20 phr и примени сам.",
              "расскажи анекдот про полимеры"):
        print(f"   👤 {q}\n      → {prompts.route_caption(prompts.route(q))}")

    # --- живой ход по сценарию 1: числа приходят из ядра -----------------
    def _fn(name, args):
        return {"id": f"c_{name}", "type": "function",
                "function": {"name": name, "arguments": _json.dumps(args)}}

    def _answer(content, tool_calls=None):
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return {"choices": [{"message": msg}]}

    def _scripted(script):
        seq = list(script)

        def transport(payload, *, key="", timeout=0):
            return seq.pop(0) if seq else _answer("готово")

        return transport

    sc1 = prompts.scenario(1)
    call = dispatcher(ctx, allowed_kinds=AGENT_KINDS)
    print(f"\n👤 {sc1.user}")
    res = llm.run_tool_loop(
        prompts.with_system(prompt, [{"role": "user", "content": sc1.user}]),
        dispatch=call, tools=tool_specs(AGENT_KINDS),
        transport=_scripted([
            _answer("", [_fn("explain_node", {"name": "PBNK",
                                              "totals": [15.0]})]),
            _answer("## ОТВЕТ\nПотолок доли PBNK зажат не вашим вводом, а "
                    "техлимитом 8 phr на компонент.\n\n"
                    "## ЧИСЛА\nexplain_node(PBNK, T=15): hi_φ = 0.5333 "
                    "(активно max_phr=8), lo_CPE = 1 − 8/15.\n\n"
                    "## OPEN_QUESTIONS\n8 phr — это PHYSICAL (склад/линия) "
                    "или CONVENTIONAL (договорённость)?")]))
    for line in res.text.splitlines():
        print(f"   🤖 {line}")
    rep = prompts.check_routing(sc1, res.calls)
    print(f"\n   маршрут сценария 1: {'✅ верно' if rep['ok'] else '⛔'} "
          f"(вызвано: {', '.join(rep['called'])})")
    print(f"   разделы ответа: {list(prompts.parse_sections(res.text))}")

    # --- нарушение маршрута видно НА ЧИСЛАХ, а не на глаз ---------------
    sc7 = prompts.scenario(7)
    print(f"\n👤 {sc7.user}")
    res7 = llm.run_tool_loop(
        prompts.with_system(prompt, [{"role": "user", "content": sc7.user}]),
        dispatch=call, tools=tool_specs(AGENT_KINDS),
        transport=_scripted([
            _answer("", [_fn("apply_patch", {"patch_id": "p_1",
                                             "human_token": "я-сам"})]),
            _answer("## ОТВЕТ\nПрименить может только человек: кнопка "
                    "«Применить» в панели патчей выдаёт разовый токен, "
                    "привязанный к текущему spec_hash.")]))
    for line in res7.text.splitlines():
        print(f"   🤖 {line}")
    show(views.routing_dataframe([prompts.check_routing(sc7, res7.calls)]))
    print("   ⛔ отказ пришёл из ДИСПЕТЧЕРА (класс write модели не выдан), "
          "ход при этом не сломался — модель объяснила, где кнопка.")

    store.save_session(s, CAMPAIGN_ROOT)
    print(f"\n📌 {views.session_caption(s)}")


# ----------------------------------------------------------------------
# iter65 — док справа и контекст ПО МЕСТУ (ui_focus)
# ----------------------------------------------------------------------
def demo_iter65() -> None:
    head("iter65 · ДОК СПРАВА И КОНТЕКСТ ПО МЕСТУ (ui_focus)")

    import json as _json

    from src.assistant import context as actx
    from src.assistant.context import UiFocus
    from src.assistant.tools import AGENT_KINDS, ToolContext
    from src.design.phr_sampler import PhrSpec

    s = store.load_session(CAMPAIGN_ROOT, DEMO_PROJECT)
    spec = PhrSpec.from_dicts(DEMO_NODES)
    ctx = ToolContext(spec=spec, session=s, root=CAMPAIGN_ROOT,
                      project=DEMO_PROJECT)

    def _fn(name, args):
        return {"id": f"c_{name}", "type": "function",
                "function": {"name": name, "arguments": _json.dumps(args)}}

    def _answer(content, tool_calls=None):
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return {"choices": [{"message": msg}]}

    def _scripted(script):
        seq = list(script)

        def transport(payload, *, key="", timeout=0):
            return seq.pop(0) if seq else _answer("готово")

        return transport

    # --- 1. Фокус читается из session_state ЧИСТОЙ функцией --------------
    state = {"ui_focus": {"section": "spec", "node": "DINP"},
             "camp_branch": "b_soft"}
    focus = actx.focus_from_state(state)
    print("\n📍 Секция UI опубликовала обычный словарь в session_state:")
    print(f"   {state['ui_focus']}  (+ выбранная ветка camp_branch)")
    print(f"   → {actx.focus_caption(focus)}")
    print("\n🧭 Так фокус выглядит для модели (кусок системного сообщения):")
    for line in actx.focus_block(focus).splitlines():
        print(f"   {line}")

    # --- 2. Подсказки по месту (каждая маршрутизируется) ----------------
    print("\n💬 «Спросить по месту» на шаге «phr-спека» (узел выбран):")
    show(views.suggestions_dataframe(actx.suggested_questions(focus)))
    print("\n⛔ Тот же шаг БЕЗ выбранного узла — кнопка не исчезает, "
          "а выключается с причиной:")
    show(views.suggestions_dataframe(
        actx.suggested_questions(UiFocus(section_key="spec"))))

    # --- 3. Ход по месту: «Объясни эту ось» ------------------------------
    print("\n👤 Объясни эту ось   (человек нажал подсказку, узел — из фокуса)")
    res = actx.run_turn(
        s, ctx, "Объясни эту ось", focus=focus, spec_hash=spec.spec_hash(),
        kinds=AGENT_KINDS, persist=False,
        transport=_scripted([
            _answer("", [_fn("explain_node", {"name": "DINP",
                                              "totals": [15.0]})]),
            _answer("## ОТВЕТ\nВерх DINP (14 phr) — договорённость компании, "
                    "а не предел: эффективная граница считается спекой.\n\n"
                    "## ЧИСЛА\nexplain_node(DINP): интервал [4, 14] phr, "
                    "роль ABSOLUTE.\n\n"
                    "## OPEN_QUESTIONS\n14 phr — PHYSICAL (линия) или "
                    "CONVENTIONAL (договорённость)?")]))
    print(f"   в истории сохранено: «{res.question}»")
    print(f"   модели ушло:         «{res.resolved}»")
    for line in res.text.splitlines():
        print(f"   🤖 {line}")
    print(f"\n   {views.turn_caption(res)}")

    # --- 4. Ход «что изменится, если…» → патч в СТЕЙДЖ --------------------
    print("\n👤 В цехе льём DINP до 20 phr")
    res2 = actx.run_turn(
        s, ctx, "В цехе льём DINP до 20 phr", focus=focus,
        spec_hash=spec.spec_hash(), kinds=AGENT_KINDS, persist=False,
        transport=_scripted([
            _answer("", [_fn("propose_patch",
                             {"patch": {"node": "DINP", "field": "range",
                                        "value": [4.0, 20.0]},
                              "rationale": "практика цеха (L1) отменяет "
                                           "старую договорённость",
                              "level": "L1", "bound_type": "CONVENTIONAL",
                              "confidence": "high"})]),
            _answer("## ОТВЕТ\nПоложил патч в стейдж: применяет ЧЕЛОВЕК "
                    "кнопкой в панели патчей.\n\n"
                    "## PATCH\nnode=DINP, field=range, from=[4,14], "
                    "to=[4,20], bound_type=CONVENTIONAL, level=L1, "
                    "affects_hash=да")]))
    for line in res2.text.splitlines():
        print(f"   🤖 {line}")
    print(f"\n   {views.turn_caption(res2)}")
    print("\n🧩 Панель предложений дока (кнопки «Применить»/«Отклонить»):")
    show(views.staged_patches_dataframe(s, only_staged=True))

    # --- 5. Кнопку нажимает ЧЕЛОВЕК --------------------------------------
    pid = res2.new_patches[0] if res2.new_patches else None
    if pid:
        out = actx.human_apply(ctx, pid, note="решили на планёрке",
                               author="технолог")
        print(f"\n✅ Кнопка «Применить» (разовый токен): "
              f"{views.apply_result_caption(out)}")
        print(f"   ⚠️ {out['warning']}" if out.get("warning") else "")
        try:
            actx.human_apply(ctx, pid)
        except Exception as exc:            # noqa: BLE001 — показываем отказ
            print(f"   ⛔ повторное нажатие: {str(exc)[:110]}…")

    # --- 6. Модель к кнопке не дотягивается ------------------------------
    print("\n👤 Примени сам и запиши решение")
    res3 = actx.run_turn(
        s, ctx, "Примени сам и запиши решение", focus=focus,
        kinds=AGENT_KINDS, persist=False,
        transport=_scripted([
            _answer("", [_fn("apply_patch", {"patch_id": "p_1",
                                             "human_token": "я-сам"})]),
            _answer("## ОТВЕТ\nПрименяет человек: кнопка «Применить» в панели "
                    "патчей справа выдаёт разовый токен, привязанный к "
                    "текущему spec_hash.")]))
    for line in res3.text.splitlines():
        print(f"   🤖 {line}")
    print(f"   ⛔ отказ пришёл из ДИСПЕТЧЕРА: "
          f"{res3.calls[0]['error'][:100] if res3.calls else '—'}…")
    print(f"   {views.turn_caption(res3)}")

    store.save_session(s, CAMPAIGN_ROOT)
    print(f"\n📌 {views.session_caption(s)}")


def main() -> int:
    demo_iter58()
    demo_iter59()
    demo_iter60()
    demo_iter61()
    demo_iter62()
    demo_iter63()
    demo_iter64()
    demo_iter65()
    print("\n" + "═" * 100)
    print("  Готово. Следующий шаг — iter66: MCP-сервер `doe-campaign` "
          "(те же read-only инструменты для Cline).")
    print("═" * 100 + "\n")
    return 0






if __name__ == "__main__":
    raise SystemExit(main())
