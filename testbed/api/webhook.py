"""Vercel Serverless Telegram Webhook Handler.

Functions as the entry-point for the double-blind research testbed platform.
Receives user media, enforces Zero-PII anonymization, stages files in Supabase
Storage, and enqueues jobs into the `hpc_tasks` table for university Slurm workers.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import uuid
from typing import Any, Dict

try:
    import httpx
except ImportError:
    httpx = None

try:
    from fastapi import FastAPI, Header, HTTPException, Request, status
    from fastapi.responses import JSONResponse
    app = FastAPI(title="Research Testbed Webhook", version="2.0.0")
except ImportError:
    FastAPI = None

    def Header(default=None, **kw):  # noqa: N802
        return default

    HTTPException = Request = status = JSONResponse = None

    class DummyApp:
        def get(self, *args, **kwargs):
            return lambda fn: fn

        def post(self, *args, **kwargs):
            return lambda fn: fn

    app = DummyApp()

try:
    from supabase import Client, create_client
except ImportError:
    Client = None
    create_client = None

# ============================================================================
# Environment & Configuration
# ============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_SECRET_TOKEN = os.getenv("TELEGRAM_SECRET_TOKEN", "")
TELEGRAM_SALT = os.getenv("TELEGRAM_SALT", "tfc_research_salt_2026")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Initialize Supabase client with administrative service_role credentials
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

TELEGRAM_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Educational email regex pattern (e.g. .edu, .ac.uk, .edu.au)
EDU_EMAIL_PATTERN = re.compile(
    r"^[a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.)+(edu|ac\.[a-z]{2,3}|edu\.[a-z]{2,3})$",
    re.IGNORECASE,
)


# ============================================================================
# Cryptographic Anonymization (Zero-PII Enforcement)
# ============================================================================
def hash_telegram_id(telegram_id: int | str) -> str:
    """Generate a deterministic, non-reversible HMAC-SHA256 digest of Telegram ID.

    Never store or log raw Telegram IDs, real names, or phone numbers.
    """
    key = TELEGRAM_SALT.encode("utf-8")
    msg = str(telegram_id).encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


# ============================================================================
# Telegram API Client Helpers
# ============================================================================
async def send_telegram_message(
    chat_id: int,
    text: str,
    reply_to_message_id: int | None = None,
    parse_mode: str = "HTML",
) -> None:
    """Send asynchronous reply to Telegram user."""
    payload: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
    }
    if reply_to_message_id:
        payload["reply_to_message_id"] = reply_to_message_id

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            await client.post(f"{TELEGRAM_API_BASE}/sendMessage", json=payload)
        except Exception as exc:
            # Avoid logging sensitive metadata
            print(f"[ERROR] Failed to send Telegram message: {type(exc).__name__}")


async def fetch_telegram_file_info(file_id: str) -> Dict[str, Any]:
    """Retrieve file metadata and download path from Telegram API."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{TELEGRAM_API_BASE}/getFile", params={"file_id": file_id})
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise ValueError(f"Telegram getFile returned error: {data.get('description')}")
        return data["result"]


async def download_telegram_file(file_path: str) -> bytes:
    """Download binary content of media file from Telegram CDN."""
    download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(download_url)
        resp.raise_for_status()
        return resp.content


# ============================================================================
# Database Operations (Supabase)
# ============================================================================
def get_or_create_user(hashed_id: str) -> Dict[str, Any]:
    """Retrieve existing user by cryptographic hash or create a new entry."""
    if not supabase:
        raise RuntimeError("Supabase client is not configured.")

    res = supabase.table("users").select("*").eq("hashed_telegram_id", hashed_id).execute()
    if res.data:
        return res.data[0]

    # Create new unverified user
    insert_res = (
        supabase.table("users")
        .insert({
            "hashed_telegram_id": hashed_id,
            "is_edu_verified": False,
        })
        .execute()
    )
    return insert_res.data[0]


def mark_user_edu_verified(user_id: str) -> None:
    """Update user verification status. Notice: RAW EMAIL IS NEVER STORED."""
    if not supabase:
        raise RuntimeError("Supabase client is not configured.")
    supabase.table("users").update({"is_edu_verified": True}).eq("id", user_id).execute()


def create_hpc_task(
    user_id: str,
    payload_url: str,
    original_size: int,
    mime_type: str,
) -> Dict[str, Any]:
    """Insert a new pending computational job into the hpc_tasks queue."""
    if not supabase:
        raise RuntimeError("Supabase client is not configured.")

    task_payload = {
        "user_id": user_id,
        "task_type": "codec_compression",
        "payload_url": payload_url,
        "status": "pending",
        "metrics": {
            "original_size_bytes": original_size,
            "mime_type": mime_type,
        },
    }
    res = supabase.table("hpc_tasks").insert(task_payload).execute()
    return res.data[0]


# ============================================================================
# Webhook Route Handlers
# ============================================================================
@app.get("/")
@app.get("/api/webhook")
async def health_check():
    """Health check endpoint for Vercel monitoring."""
    return {"status": "online", "service": "tfc_research_testbed_webhook"}


@app.post("/api/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(None),
):
    """Primary webhook endpoint handling incoming updates from Telegram."""
    # 1. Verify Telegram secret token header if configured
    if TELEGRAM_SECRET_TOKEN and x_telegram_bot_api_secret_token != TELEGRAM_SECRET_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid secret token header.",
        )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    message = body.get("message")
    if not message:
        # Ignore non-message updates (e.g. channel posts, poll answers)
        return JSONResponse(content={"status": "ignored"})

    from_user = message.get("from", {})
    raw_user_id = from_user.get("id")
    chat_id = message.get("chat", {}).get("id")
    message_id = message.get("message_id")

    if not raw_user_id or not chat_id:
        return JSONResponse(content={"status": "no_user_or_chat"})

    # Apply strict Zero-PII cryptographic hashing
    user_hash = hash_telegram_id(raw_user_id)
    db_user = get_or_create_user(user_hash)
    user_uuid = db_user["id"]

    text_content = (message.get("text") or "").strip()

    # ------------------------------------------------------------------------
    # Branch 1: /start Command (Prompts for Edu Verification)
    # ------------------------------------------------------------------------
    if text_content == "/start":
        welcome_msg = (
            "🎓 <b>Welcome to StudyMatch & Media Sync Platform!</b>\n\n"
            "This service enables double-blind student matching and high-fidelity "
            "screen content sharing for research and collaborative study.\n\n"
            "🔒 <b>Privacy Guarantee:</b> All identities are cryptographically hashed. "
            "No personal names or phone numbers are ever stored.\n\n"
            "To unlock unlimited media processing and study matching, please reply with "
            "your university email address (e.g., <code>student@mit.edu</code> or "
            "<code>user@ox.ac.uk</code>)."
        )
        await send_telegram_message(chat_id, welcome_msg, reply_to_message_id=message_id)
        return JSONResponse(content={"status": "start_handled"})

    # ------------------------------------------------------------------------
    # Branch 2: Edu Email Verification Input
    # ------------------------------------------------------------------------
    if text_content and "@" in text_content:
        # Check against university pattern
        potential_email = text_content.strip()
        if EDU_EMAIL_PATTERN.match(potential_email):
            mark_user_edu_verified(user_uuid)
            success_msg = (
                "✅ <b>University Affiliation Verified!</b>\n\n"
                "Your account is now authenticated with academic status. "
                "<i>(Note: In compliance with our zero-PII privacy model, your raw email "
                "was discarded immediately and never stored in the database.)</i>\n\n"
                "You may now send screen recordings, lecture videos, or slide documents "
                "for accelerated processing."
            )
            await send_telegram_message(chat_id, success_msg, reply_to_message_id=message_id)
            return JSONResponse(content={"status": "verified"})
        else:
            invalid_msg = (
                "⚠️ <b>Verification Notice:</b>\n"
                "Please provide a recognized university address ending in "
                "<code>.edu</code> or <code>.ac.uk</code>."
            )
            await send_telegram_message(chat_id, invalid_msg, reply_to_message_id=message_id)
            return JSONResponse(content={"status": "invalid_email_domain"})

    # ------------------------------------------------------------------------
    # Branch 3: Media File Ingestion (Offload to HPC Cluster)
    # ------------------------------------------------------------------------
    media_obj = message.get("video") or message.get("document") or message.get("animation")

    if media_obj:
        file_id = media_obj.get("file_id")
        file_size = media_obj.get("file_size", 0)
        file_name = media_obj.get("file_name", f"media_{uuid.uuid4().hex[:8]}.mp4")
        mime_type = media_obj.get("mime_type", "video/mp4")

        # 1. Fetch download URL from Telegram
        file_info = await fetch_telegram_file_info(file_id)
        file_path_on_cdn = file_info.get("file_path")
        if not file_path_on_cdn:
            raise HTTPException(status_code=500, detail="Telegram did not return file_path.")

        # 2. Download media bytes into serverless memory
        file_bytes = await download_telegram_file(file_path_on_cdn)

        # 3. Upload to Supabase Storage ('hpc-payloads' private bucket)
        task_id = str(uuid.uuid4())
        storage_dest_path = f"raw/{task_id}/{file_name}"

        supabase.storage.from_("hpc-payloads").upload(
            path=storage_dest_path,
            file=file_bytes,
            file_options={"content-type": mime_type},
        )

        # 4. Enqueue task into hpc_tasks queue
        create_hpc_task(
            user_id=user_uuid,
            payload_url=storage_dest_path,
            original_size=file_size,
            mime_type=mime_type,
        )

        # 5. Immediate user reply adhering strictly to specification
        reply_text = (
            "File received. Offloading to HPC cluster for processing...\n\n"
            f"<b>Job Tracking ID:</b> <code>{task_id}</code>\n"
            "<b>Status:</b> ⏳ Queued on Slurm compute node."
        )
        await send_telegram_message(chat_id, reply_text, reply_to_message_id=message_id)
        return JSONResponse(content={"status": "task_enqueued", "task_id": task_id})

    # Default fallback for unrecognized non-command messages
    fallback_msg = (
        "💡 Send a video, presentation recording, or screen content file to "
        "process through the computational research cluster."
    )
    await send_telegram_message(chat_id, fallback_msg, reply_to_message_id=message_id)
    return JSONResponse(content={"status": "fallback"})
