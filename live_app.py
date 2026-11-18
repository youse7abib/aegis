
import time
import html
import cv2
import torch
import gradio as gr
import pandas as pd
import os
import json
import logging
import urllib.request
import urllib.parse

from src.detector import ObjectDetector
from src.pose_analyzer import PoseAnalyzer
from src.tracker import StudentTracker
from src.rules_engine import ExamRulesEngine
from src.pipeline import SurveillancePipeline

from queue import Queue
from threading import Thread

logging.basicConfig(level=logging.INFO)

COLUMNS = [
    "Timestamp", "Student ID",
    "Violation", "Severity", "Confidence"
]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the original project models once.
detector = ObjectDetector(
    model_name="yolov8n.pt",
    conf_thresh=0.35
)
pose_analyzer = PoseAnalyzer(
    model_name="yolov8n-pose.pt"
)

detector.model.to(device)
pose_analyzer.model.to(device)

renderer = SurveillancePipeline(
    obj_detector=detector,
    pose_analyzer=pose_analyzer
)


class AlertManager:
    def __init__(self):
        self.token = os.getenv("AEGIS_TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("AEGIS_TELEGRAM_CHAT_ID")
        self.queue = Queue()

        if self.token and self.chat_id:
            Thread(
                target=self._worker,
                daemon=True
            ).start()
            logging.info("Telegram alerts enabled.")
        else:
            logging.info(
                "Telegram not configured; alerts will be logged locally."
            )

    def notify(self, incident):
        message = (
            "AEGIS — Incident Alert\n"
            f"Student ID: {incident.get('student_id', 'Unknown')}\n"
            f"Violation: {incident.get('violation', 'Unknown')}\n"
            f"Severity: {incident.get('severity', 'Unknown')}\n"
            f"Confidence: {incident.get('confidence', 'N/A')}\n"
            f"Timestamp: {incident.get('timestamp', 'N/A')}\n"
            "\nThis incident requires human review."
        )

        if not self.token or not self.chat_id:
            logging.warning("ALERT: %s", message)
            return

        self.queue.put(message)

    def _worker(self):
        while True:
            message = self.queue.get()

            try:
                url = (
                    f"https://api.telegram.org/"
                    f"bot{self.token}/sendMessage"
                )

                payload = urllib.parse.urlencode({
                    "chat_id": self.chat_id,
                    "text": message
                }).encode("utf-8")

                request = urllib.request.Request(
                    url,
                    data=payload,
                    method="POST"
                )

                with urllib.request.urlopen(
                    request, timeout=5
                ) as response:
                    result = json.loads(
                        response.read().decode("utf-8")
                    )

                if not result.get("ok"):
                    logging.error("Telegram rejected the alert.")

            except Exception:
                logging.exception("Failed to send Telegram alert.")

            finally:
                self.queue.task_done()


alert_manager = AlertManager()

def format_kpi_cards(stats=None):
    if stats is None:
        stats = {
            "total_students": "0",
            "flagged_incidents": "0",
            "clean_students": "0",
            "integrity_score": "--%"
        }

    cards = [
        ("CANDIDATES MONITORED", stats["total_students"], "#0f172a"),
        ("FLAGGED VIOLATIONS", stats["flagged_incidents"], "#dc2626"),
        ("CLEAN CANDIDATES", stats["clean_students"], "#059669"),
        ("INTEGRITY INDEX", stats["integrity_score"], "#2563eb")
    ]

    items = ""
    for label, value, color in cards:
        items += f"""
        <div style="background:white;border:1px solid #e2e8f0;
                    border-radius:10px;padding:16px;min-width:0">
          <div style="font-size:11px;font-weight:600;color:#64748b">
            {html.escape(str(label))}
          </div>
          <div style="font-size:28px;font-weight:700;
                      color:{color};margin-top:6px">
            {html.escape(str(value))}
          </div>
        </div>
        """

    return f"""
    <div style="display:grid;
                grid-template-columns:repeat(4,minmax(0,1fr));
                gap:10px;margin:10px 0">
      {items}
    </div>
    """


def empty_incident_table():
    return pd.DataFrame(columns=COLUMNS)


def new_session(mode):
    pose_analyzer.reset_baselines()

    return {
        "mode": mode,
        "started": time.monotonic(),
        "frames": 0,
        "alert_count": 0,
        "tracker": StudentTracker(
            max_distance=160.0,
            init_window_sec=2.5
        ),
        "rules": ExamRulesEngine(
            mode=mode,
            fps=10
        )
    }


def make_dashboard_stats(session):
    incidents = session["rules"].confirmed_incidents

    # Match the original pipeline's KPI definitions.
    total_students = len(session["tracker"].tracks)
    flagged_students = len(set(
        incident["student_id"] for incident in incidents
    ))
    clean_students = max(
        0, total_students - flagged_students
    )
    integrity = int(
        clean_students / max(1, total_students) * 100
    )

    stats = {
        "total_students": total_students,
        "flagged_incidents": len(incidents),
        "clean_students": clean_students,
        "integrity_score": f"{integrity}%"
    }

    return stats, incidents


def format_incident_table(incidents):
    # Explicitly map the rules engine's lowercase keys
    # to the dashboard's display column names.
    rows = [
        {
            "Timestamp": item.get("timestamp", ""),
            "Student ID": item.get("student_id", ""),
            "Violation": item.get("violation", ""),
            "Severity": item.get("severity", ""),
            "Confidence": item.get("confidence", "")
        }
        for item in incidents[-30:]
    ]

    return pd.DataFrame(rows, columns=COLUMNS)


def process_live(frame, mode, session):
    if frame is None:
        count = session.get("alert_count", 0) if session else 0

        return (
            None,
            format_kpi_cards(),
            empty_incident_table(),
            "Waiting for webcam frames...",
            session,
            f'<div data-aegis-alert-count="{count}"></div>'
        )

    if session is None or session["mode"] != mode:
        session = new_session(mode)

    # Gradio sends RGB; OpenCV and the original pipeline use BGR.
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Resize large frames to improve throughput.
    height, width = frame_bgr.shape[:2]
    if width > 960:
        new_height = int(height * 960 / width)
        frame_bgr = cv2.resize(
            frame_bgr, (960, new_height)
        )

    timestamp = time.monotonic() - session["started"]
    session["frames"] += 1

    person_boxes, prohibited_objects = detector.detect(frame_bgr)

    students = session["tracker"].update(
        person_boxes,
        current_timestamp=timestamp
    )

    student_objects = detector.match_objects_to_students(
        students, prohibited_objects
    )

    poses = pose_analyzer.analyze(frame_bgr, students)

    statuses, new_incidents = session["rules"].evaluate_frame(
        timestamp, students, poses, student_objects
    )

    for incident in new_incidents:
      alert_manager.notify(incident)


# Notify the browser when a new incident appears.
    if new_incidents:
      session["alert_count"] = session.get("alert_count", 0) + 1
    
    renderer._render_overlay(
        frame_bgr,
        students,
        prohibited_objects,
        statuses,
        timestamp,
        mode
    )

    # Update the original dashboard metrics on every processed frame.
    stats, incidents = make_dashboard_stats(session)
    kpi_html = format_kpi_cards(stats)
    incident_df = format_incident_table(incidents)

    status_text = (
        f"LIVE | {timestamp:.1f}s | "
        f"Frames processed: {session['frames']} | "
        f"Tracked candidates: {stats['total_students']} | "
        f"Flagged violations: {stats['flagged_incidents']} | "
        f"Device: {device}"
    )

    annotated_rgb = cv2.cvtColor(
        frame_bgr, cv2.COLOR_BGR2RGB
    )

    return (
        annotated_rgb,
        kpi_html,
        incident_df,
        status_text,
        session,
        f'<div data-aegis-alert-count="{session["alert_count"]}"></div>'

    )


def reset_session_ui():
    pose_analyzer.reset_baselines()

    return (
        None,
        format_kpi_cards(),
        empty_incident_table(),
        "Session reset. Start the webcam stream again.",
        '<div data-aegis-alert-count="0"></div>'

    )


with gr.Blocks(
    title="AEGIS Live Exam Surveillance",
    js="""
    () => {
        let audioContext = null;

        document.addEventListener("click", () => {
            if (!audioContext) {
                const AudioContextClass =
                    window.AudioContext || window.webkitAudioContext;

                if (AudioContextClass) {
                    audioContext = new AudioContextClass();
                }
            }

            if (audioContext && audioContext.state === "suspended") {
                audioContext.resume();
            }
        }, { once: true });

        window.aegisPlayAlert = () => {
            if (!audioContext) {
                const AudioContextClass =
                    window.AudioContext || window.webkitAudioContext;

                if (AudioContextClass) {
                    audioContext = new AudioContextClass();
                }
            }

            if (!audioContext) return;

            if (audioContext.state === "suspended") {
                audioContext.resume();
            }

            const oscillator = audioContext.createOscillator();
            const gain = audioContext.createGain();

            oscillator.type = "sine";
            oscillator.frequency.value = 880;
            gain.gain.value = 0.15;

            oscillator.connect(gain);
            gain.connect(audioContext.destination);

            oscillator.start();
            oscillator.stop(audioContext.currentTime + 0.25);
        };
        
        let lastAlertCount = 0;

        const alertObserver = new MutationObserver(() => {
            const signal = document.querySelector(
                "[data-aegis-alert-count]"
            );

            if (!signal) return;

            const count = Number(
                signal.getAttribute("data-aegis-alert-count")
            );

            if (!Number.isFinite(count)) return;

            if (count !== lastAlertCount) {
                const isNewAlert = count > lastAlertCount;
                lastAlertCount = count;

                if (isNewAlert) {
                    window.aegisPlayAlert();
                }
    }
});

alertObserver.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["data-aegis-alert-count"]
});
    }
    """
) as demo:
    gr.Markdown(
        "# AEGIS // LIVE EXAM SURVEILLANCE\n"
        "Live camera analysis with the original rules engine "
        "and dashboard KPI definitions."
    )

    mode = gr.Radio(
        choices=[
            "Classroom / Exam Hall (CCTV)",
            "Remote Online Exam (Webcam)"
        ],
        value="Classroom / Exam Hall (CCTV)",
        label="Monitoring mode"
    )

    gr.Markdown("### Dashboard")
    kpi_display = gr.HTML(value=format_kpi_cards())
    alert_signal = gr.HTML(
      value='<div data-aegis-alert-count="0"></div>',
      visible=True
      )

    with gr.Row():
        webcam = gr.Image(
            sources=["webcam"],
            type="numpy",
            label="Webcam input — start streaming"
        )

        annotated = gr.Image(
            label="Live AI output"
        )

    status = gr.Textbox(
        label="System status",
        value="Waiting for webcam stream...",
        interactive=False
    )

    incident_table = gr.Dataframe(
        headers=COLUMNS,
        datatype=["str", "str", "str", "str", "str"],
        value=empty_incident_table(),
        label="Incident Log — latest 30 entries",
        interactive=False
    )

    session_state = gr.State(value=None)

    reset_btn = gr.Button("Reset monitoring session")

    webcam.stream(
        fn=process_live,
        inputs=[webcam, mode, session_state],
        outputs=[
            annotated,
            kpi_display,
            incident_table,
            status,
            session_state,
            alert_signal
        ],
        stream_every=0.5,
        time_limit=600,
        concurrency_limit=1
    )

    reset_btn.click(
        fn=reset_session_ui,
        inputs=[],
        outputs=[
            session_state,
            kpi_display,
            incident_table,
            status,
            alert_signal
        ]
    )


if __name__ == "__main__":
    demo.queue().launch(
        share=True,
        debug=True
    )
