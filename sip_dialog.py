"""Общая диалоговая логика голосовых трактов (AudioSocket-мост и SIP-регистрация).

Оба транспорта (`sip_bridge` — Asterisk AudioSocket, `sip_phone` — нативный SIP через
pyVoIP) вели один и тот же диалог: VAD по RMS/тишине → накопление реплики → STT →
ветвление голосовой обратной связи (оценка good/bad, запрос/приём комментария) →
ответ RAG → озвучка. Раньше эти ~120 строк были продублированы в обоих модулях, и
внесённые фиксы (эхо-дрейн по «стенным часам», семафор `activity.heavy_slot` на
тяжёлых стадиях, лимиты) приходилось повторять дважды. Здесь они собраны в одном месте.

Транспорт-специфика (как читать/играть кадры, AudioSocket vs RTP, конвертация u8↔s16,
детект отбоя, бип) остаётся в модулях транспорта и подключается сюда через колбэки:

    stt(pcm16)            -> str            распознать 16-бит PCM (общий sip_bridge._stt)
    answer(question)      -> (text, rid)    ответ RAG + журнал (caller «зашит» в колбэк)
    feedback_intent(q)    -> str            'good'|'bad'|'comment'|'ask'
    rms(pcm16)            -> float          громкость кадра (общий sip_bridge._rms)
    speak(text, *, beep=False, is_answer=False) -> bool
        Озвучить и проиграть фразу средствами транспорта. Возвращает True, если во
        время проигрывания/дрейна обнаружен отбой (тогда диалог завершается). Для
        `is_answer=True` транспорт выполняет свой эхо-дрейн (у AudioSocket — по
        wall-clock с ловлей KIND_HANGUP). `beep` — доиграть сигнал (только sip_phone).

Опциональные хуки (диагностика/активность), по умолчанию no-op:
    set_stage(stage, detail=None)   обновить стадию activity
    on_frame(rms)                   на каждый кадр (пиковый RMS в SIP_DEBUG)
    on_stt(pcm, text)               после распознавания (лог в SIP_DEBUG)
    on_answer(ans, rid, speak_text) после получения ответа (лог в SIP_DEBUG)
    on_error(where, exc)            ошибка записи оценки/комментария в БД
"""
from __future__ import annotations
import time


class DialogSession:
    """Состояние и логика одного голосового диалога, независимые от транспорта.

    Транспорт в своём цикле делает:
        sess = DialogSession(cfg=..., rms=..., stt=..., answer=..., feedback_intent=...,
                             speak=...)
        while <звонок активен>:
            pcm16 = <прочитать кадр и привести к 16-бит>
            utter = sess.feed(pcm16)
            if utter is not None and sess.process(utter):
                break     # обнаружен отбой во время озвучки ответа
    """

    def __init__(self, *, cfg, rms, stt, answer, feedback_intent, speak,
                 set_stage=None, on_frame=None, on_stt=None, on_answer=None,
                 on_error=None,
                 comment_prompt="Говорите комментарий.",
                 comment_prompt_beep=False):
        self._cfg = cfg
        self._rms = rms
        self._stt = stt
        self._answer = answer
        self._feedback_intent = feedback_intent
        self._speak = speak
        self._set_stage = set_stage
        self._on_frame = on_frame
        self._on_stt = on_stt
        self._on_answer = on_answer
        self._on_error = on_error
        self.comment_prompt = comment_prompt
        self.comment_prompt_beep = comment_prompt_beep

        # Настройки VAD (единый источник для обоих трактов).
        self.silence_ms = int(cfg("SIP_SILENCE_MS", 700))
        self.silence_rms = float(cfg("SIP_SILENCE_RMS", 500))
        self.max_utter = float(cfg("SIP_MAX_UTTER_SEC", 15))
        self.need_silence = max(1, self.silence_ms // 20)   # кадр = 20 мс

        # Состояние накопления реплики.
        self.buf = bytearray()
        self.voiced = False
        self.sil = 0
        self.utter_started = 0.0
        # Состояние голосовой обратной связи по последнему ответу.
        self.last_rid = 0
        self.await_comment = False

    # -- VAD / накопление -------------------------------------------------
    def feed(self, pcm16: bytes):
        """Подать один кадр (16-бит PCM). Вернуть накопленную реплику (bytes), когда
        сработал детект конца речи (тишина или превышен лимит длины), иначе None."""
        if not pcm16:
            return None
        rms = self._rms(pcm16)
        if self._on_frame is not None:
            try:
                self._on_frame(rms)
            except Exception:
                pass
        if rms >= self.silence_rms:
            if not self.voiced:
                self.utter_started = time.time()
            self.voiced = True
            self.sil = 0
            self.buf += pcm16
        elif self.voiced:
            self.sil += 1
            self.buf += pcm16

        too_long = self.voiced and (time.time() - self.utter_started) > self.max_utter
        if self.voiced and (self.sil >= self.need_silence or too_long):
            pcm = bytes(self.buf)
            self.buf = bytearray()
            self.voiced = False
            self.sil = 0
            return pcm
        return None

    # -- обработка распознанной реплики -----------------------------------
    def process(self, pcm: bytes) -> bool:
        """Распознать реплику и отработать ветвление (оценка/комментарий/вопрос).
        Возвращает True, если во время озвучки ответа обнаружен отбой (транспорту
        нужно завершить диалог), иначе False."""
        self._stage("распознавание")
        q = self._stt(pcm)
        if self._on_stt is not None:
            try:
                self._on_stt(pcm, q)
            except Exception:
                pass
        if not q:
            return False

        # 1) ждём произнесённый комментарий к последнему ответу
        if self.await_comment and self.last_rid:
            try:
                import db
                db.set_comment(self.last_rid, q)
            except Exception as e:
                self._err("комментарий", e)
            self.await_comment = False
            return bool(self._speak("Комментарий сохранён. Спасибо."))

        # 2) голосовая обратная связь по предыдущему ответу
        intent = self._feedback_intent(q) if self.last_rid else "ask"
        if intent in ("good", "bad"):
            try:
                import db
                db.set_rating(self.last_rid, 1 if intent == "good" else -1)
            except Exception as e:
                self._err("оценка", e)
            return bool(self._speak("Спасибо, оценка сохранена. Скажите «добавь "
                                    "комментарий», если хотите оставить отзыв."))
        if intent == "comment":
            self.await_comment = True
            return bool(self._speak(self.comment_prompt, beep=self.comment_prompt_beep))

        # 3) обычный вопрос → ответ RAG → озвучка (с эхо-дрейном у транспорта)
        self._stage("ответ", q[:60])
        ans, self.last_rid = self._answer(q)
        if not ans:
            ans = "Извините, не нашёл ответа в документах."
        if self._cfg("SIP_SPEAK_ANSWER", True):
            speak_text = ans
        else:
            speak_text = self._cfg("SIP_ACK_PHRASE",
                                   "Ваш запрос принят и записан. Спасибо.")
        if self._on_answer is not None:
            try:
                self._on_answer(ans, self.last_rid, speak_text)
            except Exception:
                pass
        return bool(self._speak(speak_text, is_answer=True))

    # -- вспомогательное --------------------------------------------------
    def _stage(self, stage, detail=None):
        if self._set_stage is not None:
            try:
                self._set_stage(stage, detail)
            except Exception:
                pass

    def _err(self, where, exc):
        if self._on_error is not None:
            try:
                self._on_error(where, exc)
            except Exception:
                pass
