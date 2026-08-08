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


def main() -> int:
    demo_iter58()
    demo_iter59()
    demo_iter60()
    print("\n" + "═" * 100)
    print("  Готово. Следующий шаг — iter61: настоящие read-only инструменты "
          "(get_spec / explain_node / validate_spec / simulate_bounds / preflight).")
    print("═" * 100 + "\n")
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
