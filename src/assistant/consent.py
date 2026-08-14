"""assistant/consent.py — токены подтверждения ЧЕЛОВЕКОМ (iter63).

Инвариант слоя (ASSISTANT_SPEC §2): *ни один write-инструмент не меняет
состояние сам*. Модель может лишь ПРЕДЛОЖИТЬ патч; применяет его человек
кнопкой в интерфейсе. Технически «кнопка» — это разовый токен, который UI
выдаёт (:meth:`ConsentRegistry.issue`) и который write-инструмент гасит
(:meth:`ConsentRegistry.consume`).

Почему токен, а не флаг «пользователь согласен»:

* **одноразовость** — повторный вызов того же инструмента с тем же токеном не
  применит патч дважды («применено дважды» не должно выглядеть нормой);
* **привязка к действию и цели** — согласие на применение патча P1 не является
  согласием на применение P2 и тем более на запись факта в журнал; модель,
  увидевшая токен в истории диалога, не сможет им воспользоваться иначе;
* **привязка к отпечатку спеки** (``context_hash``) — если между нажатием
  кнопки и применением геометрия успела измениться (другой патч, загрузка
  проекта), токен недействителен: человек подтверждал ДРУГОЕ состояние;
* **срок жизни** — подтверждение, найденное в логах через неделю, ничего не
  применяет.

Реестр живёт в памяти процесса (в UI — в ``st.session_state``): токен не
переживает перезапуск приложения намеренно, ведь подтверждение относится к
конкретному сеансу работы человека.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

#: Действия, которые вообще можно подтвердить (белый список: неизвестное
#: действие — ошибка выдачи, а не «подтверждено на всякий случай»).
#:
#: ``apply_spec`` (iter71) — применение ПАКЕТА спеки: первичный ввод геометрии
#: и её эволюция (добавить/удалить узел, сменить роль). Отдельное действие, а
#: не разновидность ``apply_patch``: подтверждение сдвига одной границы не
#: должно годиться для замены всей спеки.
#:
#: ``apply_project`` (iter73) — принятие ПАКЕТА ПРОЕКТА: состав + отклики +
#: процесс-оси, то есть РОЖДЕНИЕ проекта, а не правка существующего. Снова
#: отдельное действие: согласие на замену геометрии не есть согласие завести
#: проект с другими откликами и осями.
#: ``apply_setup`` (iter76) — применение ТОЧЕЧНОЙ ПРАВКИ ПОЛЕЙ формы сетапа
#: несобранного проекта: до сборки данные живут в полях формы, и согласие на
#: пакет проекта целиком не есть согласие на правку отдельного поля.
#: ``apply_note`` / ``reject_note`` (iter96) — фиксация ПРЕДЛОЖЕННОЙ записи
#: журнала (решение компании или L1-факт) и отказ от неё. Отдельно от
#: ``record_decision``/``add_local_fact``, которыми пишет РУЧНАЯ форма: там
#: цель подтверждения — текст, набранный человеком, здесь — идентификатор
#: предложения помощника, поля которого человек мог поправить перед записью.
ACTIONS = ("apply_patch", "reject_patch", "record_decision", "add_local_fact",
           "apply_spec", "reject_spec", "apply_project", "reject_project",
           "apply_setup", "reject_setup", "apply_note", "reject_note")

#: Срок жизни подтверждения по умолчанию, сек.
DEFAULT_TTL_S = 600.0


class ConsentError(RuntimeError):
    """Подтверждение отсутствует/не подходит — с объяснением причины (A0.6).

    Текст уходит МОДЕЛИ как результат вызова write-инструмента, поэтому он
    формулируется как указание, что делать дальше: «нажмите кнопку», «патч
    другой», «геометрия изменилась — предложите патч заново».
    """


@dataclass
class Consent:
    """Разовое подтверждение человека на одно действие с одной целью."""
    token: str
    action: str
    target: str = ""
    context_hash: str = ""
    note: str = ""
    issued_at: float = 0.0
    ttl_s: float = DEFAULT_TTL_S
    used_at: float = 0.0

    @property
    def used(self) -> bool:
        return self.used_at > 0.0

    def expires_at(self) -> float:
        return self.issued_at + float(self.ttl_s)

    def to_state(self) -> Dict[str, Any]:
        return {"token": self.token, "action": self.action,
                "target": self.target, "context_hash": self.context_hash,
                "note": self.note, "issued_at": self.issued_at,
                "ttl_s": self.ttl_s, "used_at": self.used_at}


class ConsentRegistry:
    """Выдача и гашение подтверждений (в памяти сеанса).

    ``clock`` подменяется в тестах: проверить «токен протух» временем ожидания
    в тесте — значит сделать набор медленным и хрупким.
    """

    def __init__(self, *, ttl_s: float = DEFAULT_TTL_S,
                 clock: Optional[Callable[[], float]] = None) -> None:
        if ttl_s <= 0:
            raise ValueError("ttl_s подтверждения должен быть > 0.")
        self.ttl_s = float(ttl_s)
        self._clock = clock or time.time
        self._items: Dict[str, Consent] = {}

    # -- выдача ---------------------------------------------------------
    def issue(self, action: str, target: str = "", *, context_hash: str = "",
              ttl_s: Optional[float] = None, note: str = "") -> Consent:
        """Выдать токен (это делает КНОПКА в UI, а не модель)."""
        action = str(action)
        if action not in ACTIONS:
            raise ConsentError(
                f"Неизвестное действие для подтверждения: {action!r}. "
                f"Подтверждать можно только {list(ACTIONS)}.")
        c = Consent(token=secrets.token_urlsafe(12), action=action,
                    target=str(target or ""), context_hash=str(context_hash or ""),
                    note=str(note or ""), issued_at=float(self._clock()),
                    ttl_s=float(ttl_s if ttl_s is not None else self.ttl_s))
        self._items[c.token] = c
        return c

    # -- гашение --------------------------------------------------------
    def consume(self, token: str, *, action: str, target: str = "",
                context_hash: str = "") -> Consent:
        """Погасить токен под КОНКРЕТНОЕ действие/цель или объяснить отказ."""
        token = str(token or "")
        if not token:
            raise ConsentError(
                f"Действие '{action}' меняет состояние проекта и требует "
                f"подтверждения человека: в интерфейсе нажмите кнопку "
                f"подтверждения — она выдаст разовый токен. Сам ты применить "
                f"изменение не можешь (ASSISTANT_SPEC §2).")
        c = self._items.get(token)
        if c is None:
            raise ConsentError(
                "Токен подтверждения не найден: он выдаётся кнопкой в "
                "интерфейсе и живёт в текущем сеансе. Токен из переписки или "
                "из журнала недействителен.")
        now = float(self._clock())
        if c.used:
            raise ConsentError(
                f"Токен уже использован ({time.strftime('%H:%M:%S', time.localtime(c.used_at))}): "
                f"подтверждение ОДНОРАЗОВОЕ. Повторное применение того же "
                f"изменения — отдельное решение человека.")
        if now > c.expires_at():
            raise ConsentError(
                f"Срок действия подтверждения истёк ({c.ttl_s:.0f} с). "
                f"Подтвердите действие заново — за это время состояние проекта "
                f"могло измениться.")
        if c.action != str(action):
            raise ConsentError(
                f"Подтверждение выдано на действие '{c.action}', а вызвано "
                f"'{action}'. Согласие на одно действие не переносится на "
                f"другое.")
        if str(target or "") != c.target:
            raise ConsentError(
                f"Подтверждение выдано на цель '{c.target}', а вызов "
                f"относится к '{target}'. Человек подтверждал ДРУГОЙ объект.")
        if c.context_hash and str(context_hash or "") != c.context_hash:
            raise ConsentError(
                f"Геометрия изменилась после подтверждения: подтверждали при "
                f"spec_hash={c.context_hash[:12]}…, сейчас "
                f"{str(context_hash)[:12]}…. Патч считался от прежней спеки — "
                f"предложите его заново и подтвердите ещё раз.")
        c.used_at = now
        return c

    # -- обзор ----------------------------------------------------------
    def pending(self) -> List[Consent]:
        """Выданные, но не использованные и не протухшие подтверждения."""
        now = float(self._clock())
        return [c for c in self._items.values()
                if not c.used and now <= c.expires_at()]

    def get(self, token: str) -> Optional[Consent]:
        return self._items.get(str(token or ""))

    def revoke(self, token: str) -> bool:
        """Отозвать подтверждение (человек передумал до применения)."""
        return self._items.pop(str(token or ""), None) is not None

    def clear(self) -> None:
        self._items.clear()


#: Реестр по умолчанию: им пользуются инструменты, если в контексте вызова
#: (``ToolContext.extra['consent']``) не передан свой.
DEFAULT_REGISTRY = ConsentRegistry()


def issue_token(action: str, target: str = "", *, context_hash: str = "",
                ttl_s: Optional[float] = None, note: str = "") -> str:
    """Выдать токен в реестре по умолчанию и вернуть его строкой."""
    return DEFAULT_REGISTRY.issue(action, target, context_hash=context_hash,
                                  ttl_s=ttl_s, note=note).token
