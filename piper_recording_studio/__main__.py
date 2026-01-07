"""Run the Piper recording studio web app and manage recording outputs."""

import argparse
import asyncio
import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import uuid4

import hypercorn
from quart import (
    Quart,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

_LOGGER = logging.getLogger(__name__)
_DIR = Path(__file__).parent


@dataclass
class Prompt:
    """Single prompt for the user to read."""

    group: str
    id: str
    text: str


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    #
    parser.add_argument(
        "--prompts",
        help="Path to prompts directory",
        action="append",
    )
    parser.add_argument(
        "--output",
        help="Path to output directory",
    )
    #
    parser.add_argument(
        "--multi-user",
        action="store_true",
        help="Require login code and user output directory to exist",
    )
    parser.add_argument("--cc0", action="store_true", help="Show public domain notice")
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    prompts_dirs = [Path(p) for p in args.prompts]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    css_dir = _DIR / "css"
    js_dir = _DIR / "js"
    img_dir = _DIR / "img"
    webfonts_dir = _DIR / "webfonts"

    prompts, languages = load_prompts(prompts_dirs)
    
    app = Quart("piper", template_folder=str(_DIR / "templates"))
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 mb
    app.secret_key = str(uuid4())

    @app.route("/")
    @app.route("/index.html")
    async def api_index() -> str:
        """Main page"""
        return await render_template(
            "index.html",
            languages=sorted(languages.items()),
            multi_user=args.multi_user,
            cc0=args.cc0,
            user_error=None,
        )

    @app.route("/done.html")
    async def api_done() -> str:
        return await render_template("done.html")

    @app.route("/record")
    async def api_record() -> str:
        """Record audio for a text prompt"""
        language = request.args["language"]

        if args.multi_user:
            user_id = request.args.get("userId")
            user_dir, language_dir, exists = locate_user_language_dir(
                output_dir,
                user_id,
                language,
            )
            if not user_dir.is_dir():
                return await render_template(
                    "index.html",
                    languages=sorted(languages.items()),
                    multi_user=args.multi_user,
                    cc0=args.cc0,
                    user_error=f"Recording folder for login code '{user_id}' does not exist. Please create it first.",
                )
            if not exists:
                user_dir, language_dir = create_user_language_dir(
                    output_dir,
                    user_id,
                    language,
                )
            audio_dir = user_dir
        else:
            user_id = None
            audio_dir = output_dir

        next_prompt, num_complete, num_items = get_next_prompt(
            prompts,
            audio_dir,
            language,
        )
        if next_prompt is None:
            return await render_template("done.html")

        emotion_label = getattr(next_prompt, "emotion_label", None)
        emotion_intensity = getattr(next_prompt, "emotion_intensity", None)
        complete_percent = 100 * (num_complete / num_items if num_items > 0 else 1)
        return await render_template(
            "record.html",
            language=language,
            prompt_group=next_prompt.group,
            prompt_id=next_prompt.id,
            text=next_prompt.text,
            emotion_label=emotion_label,
            emotion_intensity=emotion_intensity,
            num_complete=num_complete,
            num_items=num_items,
            complete_percent=complete_percent,
            multi_user=args.multi_user,
            user_id=user_id,
        )

    @app.route("/submit", methods=["POST"])
    async def api_submit() -> Response:
        """Submit audio for a text prompt"""
        form = await request.form
        language = form["language"]
        prompt_group = form["promptGroup"]
        prompt_id = form["promptId"]
        prompt_text = form["text"]
        audio_format = form["format"]

        files = await request.files
        assert "audio" in files, "No audio"

        suffix = ".webm"
        if "wav" in audio_format:
            suffix = ".wav"

        if args.multi_user:
            user_id = form.get("userId")
            user_dir, language_dir, exists = locate_user_language_dir(
                output_dir,
                user_id,
                language,
            )
            if not user_dir.is_dir():
                raise ValueError("Invalid login code")
            if not exists:
                user_dir, language_dir = create_user_language_dir(
                    output_dir,
                    user_id,
                    language,
                )

            audio_dir = user_dir
        else:
            audio_dir = output_dir

        # Save audio and transcription
        audio_path = audio_dir / language / prompt_group / f"{prompt_id}{suffix}"
        _LOGGER.debug("Saving to %s", audio_path)

        audio_path.parent.mkdir(parents=True, exist_ok=True)
        await files["audio"].save(audio_path)

        text_path = audio_path.parent / f"{prompt_id}.txt"
        text_path.write_text(prompt_text, encoding="utf-8")

        # Get next prompt
        next_prompt, num_complete, num_items = get_next_prompt(
            prompts,
            audio_dir,
            language,
        )
        if next_prompt is None:
            return jsonify({"done": True})

        emotion_label = getattr(next_prompt, "emotion_label", None)
        emotion_intensity = getattr(next_prompt, "emotion_intensity", None)
        complete_percent = 100 * (num_complete / num_items if num_items > 0 else 1)
        return jsonify(
            {
                "done": False,
                "promptGroup": next_prompt.group,
                "promptId": next_prompt.id,
                "promptText": next_prompt.text,
                "emotionLabel": emotion_label,
                "emotionIntensity": emotion_intensity,
                "numComplete": num_complete,
                "numItems": num_items,
                "completePercent": complete_percent,
            }
        )

    @app.route("/skip", methods=["POST"])
    async def api_skip() -> Response:
        """Skip a text prompt without recording audio."""
        form = await request.form
        language = form["language"]
        prompt_group = form["promptGroup"]
        prompt_id = form["promptId"]
        prompt_text = form["text"]

        if args.multi_user:
            user_id = form.get("userId")
            user_dir, language_dir, exists = locate_user_language_dir(
                output_dir,
                user_id,
                language,
            )
            if not user_dir.is_dir():
                raise ValueError("Invalid login code")
            if not exists:
                user_dir, language_dir = create_user_language_dir(
                    output_dir,
                    user_id,
                    language,
                )
            audio_dir = user_dir
        else:
            audio_dir = output_dir

        skip_path = audio_dir / language / prompt_group / f"{prompt_id}.skip"
        _LOGGER.debug("Marking prompt skipped: %s", skip_path)
        skip_path.parent.mkdir(parents=True, exist_ok=True)
        skip_path.write_text(prompt_text, encoding="utf-8")

        next_prompt, num_complete, num_items = get_next_prompt(
            prompts,
            audio_dir,
            language,
        )
        if next_prompt is None:
            return jsonify({"done": True})

        emotion_label = getattr(next_prompt, "emotion_label", None)
        emotion_intensity = getattr(next_prompt, "emotion_intensity", None)
        complete_percent = 100 * (num_complete / num_items if num_items > 0 else 1)
        return jsonify(
            {
                "done": False,
                "promptGroup": next_prompt.group,
                "promptId": next_prompt.id,
                "promptText": next_prompt.text,
                "emotionLabel": emotion_label,
                "emotionIntensity": emotion_intensity,
                "numComplete": num_complete,
                "numItems": num_items,
                "completePercent": complete_percent,
            }
        )

    @app.route("/upload")
    async def api_upload() -> str:
        """Upload an existing dataset"""
        language = request.args["language"]

        if args.multi_user:
            user_id = request.args.get("userId")
            user_dir, language_dir, exists = locate_user_language_dir(
                output_dir,
                user_id,
                language,
            )
            if not user_dir.is_dir():
                raise RuntimeError("Invalid login code")
            if not exists:
                user_dir, language_dir = create_user_language_dir(
                    output_dir,
                    user_id,
                    language,
                )
        else:
            user_id = None
            user_dir = output_dir

        return await render_template(
            "upload.html",
            language=language,
            multi_user=args.multi_user,
            user_id=user_id,
        )

    @app.route("/dataset", methods=["POST"])
    async def api_dataset() -> str:
        """Upload an existing dataset"""
        form = await request.form
        language = form["language"]

        if args.multi_user:
            user_id = form.get("userId")
            user_dir, language_dir, exists = locate_user_language_dir(
                output_dir,
                user_id,
                language,
            )
            if not user_dir.is_dir():
                raise RuntimeError("Invalid login code")
            if not exists:
                user_dir, language_dir = create_user_language_dir(
                    output_dir,
                    user_id,
                    language,
                )
        else:
            user_id = None
            user_dir = output_dir
            language_dir = user_dir / language

        files = await request.files
        dataset_file = files["dataset"]
        upload_path = language_dir / "_uploads" / Path(dataset_file.filename).name
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        await dataset_file.save(upload_path)
        _LOGGER.debug("Saved dataset to %s", upload_path)

        return await render_template("done.html")

    @app.route("/create_user_dir", methods=["POST"])
    async def api_create_user_dir() -> Response:
        if not args.multi_user:
            raise RuntimeError("Multi-user mode is required for this operation")

        form = await request.form
        language = form["language"]
        user_id = form.get("userId")
        create_user_language_dir(output_dir, user_id, language)

        return jsonify({"created": True})

    @app.errorhandler(Exception)
    async def handle_error(err) -> Tuple[str, int]:
        """Return error as text."""
        _LOGGER.exception(err)
        return (f"{err.__class__.__name__}: {err}", 500)

    @app.route("/css/<path:filename>", methods=["GET"])
    async def css(filename) -> Response:
        """CSS static endpoint."""
        return await send_from_directory(css_dir, filename)

    @app.route("/js/<path:filename>", methods=["GET"])
    async def js(filename) -> Response:
        """Javascript static endpoint."""
        return await send_from_directory(js_dir, filename)

    @app.route("/img/<path:filename>", methods=["GET"])
    async def img(filename) -> Response:
        """Image static endpoint."""
        return await send_from_directory(img_dir, filename)

    @app.route("/webfonts/<path:filename>", methods=["GET"])
    async def webfonts(filename) -> Response:
        """Webfonts static endpoint."""
        return await send_from_directory(webfonts_dir, filename)

    # Run web server
    hyp_config = hypercorn.config.Config()
    hyp_config.bind = [f"{args.host}:{args.port}"]

    asyncio.run(hypercorn.asyncio.serve(app, hyp_config))


# -----------------------------------------------------------------------------

def locate_user_language_dir(output_dir, user_id, language):
    """Locate the user/language folder without mutating the filesystem.

    :param output_dir: base output path for recordings
    :param user_id: login code string
    :param language: language code string
    :returns: tuple ``(user_dir, language_dir, exists)``
    """
    if not user_id or not str(user_id).strip():
        _LOGGER.warning("Missing login code when locating user folder")
        raise ValueError("Missing login code")

    user_dir = output_dir / f"user_{user_id}"
    language_dir = user_dir / language
    exists = language_dir.is_dir()
    if not exists:
        _LOGGER.warning("User folder missing: %s", language_dir)
    return user_dir, language_dir, exists


def create_user_language_dir(output_dir, user_id, language):
    """Create the directory tree for a user and language.

    :param output_dir: base output path for recordings
    :param user_id: login code string
    :param language: language code string
    :returns: tuple ``(user_dir, language_dir)``
    """
    user_dir, language_dir, _ = locate_user_language_dir(output_dir, user_id, language)
    if user_dir is None or language_dir is None:
        raise ValueError("Invalid user ID or language")
    if not user_dir.is_dir():
        raise ValueError("User folder does not exist")
    _LOGGER.warning("Creating user directory %s", language_dir)
    language_dir.mkdir(parents=True, exist_ok=True)
    return user_dir, language_dir


def load_prompts(
    prompts_dirs: List[Path],
) -> Tuple[Dict[str, List[Prompt]], Dict[str, str]]:
    prompts = defaultdict(list)
    languages = {}

    for prompts_dir in prompts_dirs:
        for language_dir in prompts_dir.iterdir():
            if not language_dir.is_dir():
                continue

            name, code = language_dir.name.rsplit("_", maxsplit=1)
            languages[name] = code
            for prompt_path in language_dir.glob("*.txt"):
                _LOGGER.debug("Loading prompts from %s", prompt_path)
                prompt_group = prompt_path.stem
                with open(prompt_path, "r", encoding="utf-8") as prompt_file:
                    reader = csv.reader(prompt_file, delimiter="\t")
                    for i, row in enumerate(reader):
                        emotion_label = None
                        emotion_intensity = None
                        if len(row) == 1:
                            prompt_id = str(i)
                            prompt_text = row[0]
                        elif len(row) == 2:
                            prompt_id = row[0]
                            prompt_text = row[1]
                        elif len(row) == 3:
                            prompt_id = row[0]
                            prompt_text = row[1]
                            emotion_label = row[2].strip() or None
                        else:
                            prompt_id = row[0]
                            prompt_text = row[1]
                            emotion_label = row[2].strip() or None
                            emotion_intensity = row[3].strip() or None

                        prompt = Prompt(
                            group=prompt_group,
                            id=prompt_id,
                            text=prompt_text,
                        )
                        prompt.emotion_label = emotion_label
                        prompt.emotion_intensity = emotion_intensity
                        prompts[code].append(prompt)

    return prompts, languages


def get_next_prompt(
    prompts: Dict[str, List[Prompt]],
    output_dir: Path,
    language: str,
):
    language_prompts = prompts[language]
    language_dir = output_dir / language
    incomplete_prompts = []
    for prompt in language_prompts:
        text_path = language_dir / prompt.group / f"{prompt.id}.txt"
        skip_path = language_dir / prompt.group / f"{prompt.id}.skip"
        if not (text_path.exists() or skip_path.exists()):
            incomplete_prompts.append(prompt)

    num_items = len(language_prompts)
    num_complete = num_items - len(incomplete_prompts)

    if incomplete_prompts:
        next_prompt = incomplete_prompts[0]
    else:
        next_prompt = None

    return next_prompt, num_complete, num_items


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
