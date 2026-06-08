"""MinerU 文档解析器封装

使用 MinerU LocalAPIServer 子进程进行高质量文档解析，
支持 OCR、表格结构保留、公式识别，输出结构化 Markdown。
"""

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

logger = logging.getLogger(__name__)


class MinerUParser:
    """MinerU 文档解析器，通过 LocalAPIServer 子进程提供高质量文档解析"""

    # MinerU 原生支持的格式（比原有 loader 多了 PPTX 和图片）
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".png", ".jpg", ".jpeg"}

    def __init__(self):
        self.enabled = os.getenv("MINERU_ENABLED", "false").lower() == "true"
        self.backend = os.getenv("MINERU_BACKEND", "pipeline")
        self.parse_method = os.getenv("MINERU_PARSE_METHOD", "auto")
        self.lang = os.getenv("MINERU_LANG", "ch")
        self._server = None
        self._base_url = None
        self._import_error = None

    def is_supported(self, filename: str) -> bool:
        """判断文件格式是否由 MinerU 处理"""
        return Path(filename).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def start_server(self):
        """启动 MinerU API 服务器（应在 FastAPI startup 事件中调用）"""
        if not self.enabled:
            return
        try:
            from mineru.cli.api_client import LocalAPIServer, wait_for_local_api_ready
            import httpx
            from mineru.cli.api_client import build_http_timeout

            self._server = LocalAPIServer()
            self._base_url = self._server.start()
            logger.info(f"MinerU API server started at {self._base_url}")

            # 等待服务就绪
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                with httpx.AsyncClient(timeout=build_http_timeout()) as client:
                    loop.run_until_complete(
                        wait_for_local_api_ready(client, self._server)
                    )
            finally:
                loop.close()

            logger.info("MinerU API server ready")
        except ImportError as e:
            self._import_error = str(e)
            self.enabled = False
            logger.warning(f"MinerU not installed, disabling: {e}")
        except Exception as e:
            self.enabled = False
            logger.error(f"Failed to start MinerU server: {e}")

    def stop_server(self):
        """停止 MinerU API 服务器（应在 FastAPI shutdown 事件中调用）"""
        if self._server:
            try:
                self._server.stop()
                logger.info("MinerU API server stopped")
            except Exception as e:
                logger.warning(f"Error stopping MinerU server: {e}")
            self._server = None
            self._base_url = None

    def parse(self, file_path: str, filename: str) -> list[Document]:
        """
        解析文档，返回按页分段的 LangChain Document 列表。

        优先使用 content_list.json（结构化，按页分组），
        回退到 Markdown 输出。
        """
        if not self._base_url:
            raise RuntimeError("MinerU server not started")

        import requests as req

        out_dir = Path(tempfile.mkdtemp(prefix="mineru_output_"))
        try:
            # 通过 REST API 同步解析（/file_parse 端点直接返回结果）
            with open(file_path, "rb") as f:
                resp = req.post(
                    f"{self._base_url}/file_parse",
                    files={"files": (filename, f)},
                    data={
                        "return_md": "true",
                        "return_content_list": "true",
                        "return_middle_json": "false",
                        "return_model_output": "false",
                        "return_images": "false",
                        "response_format_zip": "true",
                        "return_original_file": "false",
                        "backend": self.backend,
                        "parse_method": self.parse_method,
                        "lang_list": json.dumps([self.lang]),
                        "formula_enable": "true",
                        "table_enable": "true",
                    },
                    timeout=600,
                )
                resp.raise_for_status()

            # 处理 ZIP 响应
            zip_path = out_dir / "result.zip"
            zip_path.write_bytes(resp.content)

            from mineru.cli.api_client import safe_extract_zip
            extract_dir = out_dir / "extracted"
            safe_extract_zip(zip_path, extract_dir)

            # 优先读取 content_list.json（结构化，支持按页分组）
            result_dir = self._find_result_dir(extract_dir, filename)
            docs = []
            if result_dir:
                content_list_path = result_dir / "content_list.json"
                if content_list_path.exists():
                    with open(content_list_path, "r", encoding="utf-8") as f:
                        content_list = json.load(f)
                    docs = self._extract_pages_from_content_list(content_list)

            # 回退到 Markdown
            if not docs:
                docs = self._extract_pages_from_markdown(result_dir, filename)

            if not docs:
                raise ValueError("MinerU parse produced no content")

            return docs
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def load_as_chunks(
        self, file_path: str, filename: str, split_fn
    ) -> list[dict]:
        """
        解析文档并分片，返回与 DocumentLoader.load_document() 格式一致的结果。

        :param file_path: 文件路径
        :param filename: 文件名
        :param split_fn: 三级分片函数，签名 (text, base_doc, idx) -> list[dict]
        :return: 分片后的文档列表
        """
        docs = self.parse(file_path, filename)
        result = []
        page_global_chunk_idx = 0
        for doc in docs:
            base_doc = {
                "filename": filename,
                "file_path": file_path,
                "file_type": "MinerU",
                "page_number": doc.metadata.get("page", 0),
            }
            page_chunks = split_fn(
                text=(doc.page_content or "").strip(),
                base_doc=base_doc,
                page_global_chunk_idx=page_global_chunk_idx,
            )
            page_global_chunk_idx += len(page_chunks)
            result.extend(page_chunks)
        return result

    @staticmethod
    def _find_result_dir(out_dir: Path, filename: str) -> Path | None:
        """查找 MinerU 解压后的结果目录"""
        stem = Path(filename).stem
        # MinerU 通常以文件名（无扩展名）作为子目录
        candidate = out_dir / stem
        if candidate.is_dir():
            return candidate
        # 回退：找第一个包含 .md 或 .json 文件的目录
        for d in out_dir.rglob("*"):
            if d.is_dir() and list(d.glob("*.md")):
                return d
        return out_dir

    @staticmethod
    def _extract_pages_from_content_list(content_list: list) -> list[Document]:
        """从 content_list.json 按 page_idx 提取页面文本"""
        pages: dict[int, list[str]] = {}
        for item in content_list:
            page_idx = item.get("page_idx", 0)
            text = item.get("text", "").strip()
            if text:
                pages.setdefault(page_idx, []).append(text)

        docs = []
        for page_idx in sorted(pages.keys()):
            page_text = "\n".join(pages[page_idx])
            if page_text.strip():
                docs.append(
                    Document(
                        page_content=page_text,
                        metadata={"page": page_idx},
                    )
                )
        return docs

    @staticmethod
    def _extract_pages_from_markdown(
        result_dir: Path | None, filename: str
    ) -> list[Document]:
        """从 Markdown 输出中提取文本，作为 content_list 的回退"""
        if not result_dir:
            return []

        md_files = list(result_dir.glob("*.md"))
        if not md_files:
            return []

        md_path = md_files[0]
        md_text = md_path.read_text(encoding="utf-8")
        if not md_text.strip():
            return []

        # 尝试按页面分隔符分割
        page_pattern = re.compile(r"(?:^|\n)(?:---\s*\n)?(?:Page\s+)?(\d+)\s*\n", re.IGNORECASE)
        parts = page_pattern.split(md_text)

        if len(parts) > 2:
            docs = []
            for i in range(1, len(parts), 2):
                try:
                    page_num = int(parts[i])
                except (ValueError, IndexError):
                    page_num = i // 2
                content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if content:
                    docs.append(
                        Document(page_content=content, metadata={"page": page_num})
                    )
            if docs:
                return docs

        # 无法按页分割，整体作为一个 Document
        return [Document(page_content=md_text, metadata={"page": 0})]


# 模块级单例
mineru_parser = MinerUParser()
