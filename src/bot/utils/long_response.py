"""Send long responses as .md file attachments instead of many split messages."""

import os
import tempfile
from typing import Any, List

import structlog

logger = structlog.get_logger()

# Threshold: if more than this many message parts, send as file
MAX_INLINE_PARTS = 3


async def send_long_response_as_file(
    update: Any,
    formatted_messages: List[Any],
) -> bool:
    """Send formatted messages as a .md file if they exceed MAX_INLINE_PARTS.

    Args:
        update: Telegram Update object.
        formatted_messages: List of FormattedMessage with .text attribute.

    Returns:
        True if the response was sent as a file, False otherwise
        (caller should fall back to inline sending).
    """
    if len(formatted_messages) <= MAX_INLINE_PARTS:
        return False

    full_text = "\n\n".join(m.text for m in formatted_messages if m.text)
    if not full_text.strip():
        return False

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".md",
            prefix="response_",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(full_text)
            temp_path = f.name

        with open(temp_path, "rb") as doc:
            await update.message.reply_document(
                document=doc,
                filename="response.md",
                caption="Reponse trop longue pour Telegram — fichier joint.",
                reply_to_message_id=update.message.message_id,
            )

        logger.info(
            "Long response sent as file",
            parts=len(formatted_messages),
            chars=len(full_text),
        )
        return True

    except Exception as doc_err:
        logger.warning(
            "Failed to send response as document, falling back to split messages",
            error=str(doc_err),
        )
        return False

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
