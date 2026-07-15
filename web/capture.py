"""
web/capture.py — 하네스 실행 경로 캡처.

harness.graph의 각 노드는 logging.getLogger("harness")로
"ROUTER → ...", "RETRIEVE → ..." 같은 경로를 남긴다(scripts/ask.py가 화면에
찍는 그 로그). 웹에서는 이 로그를 요청별로 캡처해 응답 JSON에 담는다.

주의(동시성): 캡처 핸들러는 프로세스 전역 로거에 붙으므로, 요청이 동시에
들어오면 경로가 섞인다. 그래서 락으로 한 번에 한 질의만 처리한다.
이 프로젝트는 단일 GPU 로컬 데모라 직렬 처리로 충분하며, 다중 사용자
서비스로 키우려면 경로 캡처를 노드 반환값 방식으로 바꾸는 게 맞다.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager

_LOCK = threading.Lock()


class _ListHandler(logging.Handler):
    def __init__(self, sink: list[str]):
        super().__init__(level=logging.INFO)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.append(record.getMessage())


@contextmanager
def capture_path():
    """
    with capture_path() as path:
        result = graph.invoke({"query": q})
    # path == ["ROUTER   → 질의 분류: retrieve", "RETRIEVE → ...", ...]
    """
    lines: list[str] = []
    logger = logging.getLogger("harness")
    handler = _ListHandler(lines)
    with _LOCK:                     # 질의 직렬화 (경로 섞임 방지)
        prev_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            yield lines
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev_level)
