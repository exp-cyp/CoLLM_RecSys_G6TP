"""Turn completed evaluation into a readable local report and source/results ZIP."""

from __future__ import annotations

import base64
import html
import json
import math
import shutil
from datetime import datetime

import yaml

from scripts.package_submission import package_project

from .common import ROOT, project_path, read_json, results_path, write_json

MODEL_NAMES = ("MF", "TALLRec", "CoLLM-MF")
STAGE_NAMES = {"mf": "MF", "stage1": "TALLRec / LoRA", "stage2": "CoLLM / MLP"}


def _duration(seconds):
    seconds = max(0, round(float(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _pipeline_seconds(metadata):
    for key in ("elapsed_seconds", "duration_seconds"):
        if isinstance(metadata.get(key), (int, float)):
            return max(0.0, float(metadata[key]))
    start = metadata.get("started_at") or metadata.get("created_at")
    end = metadata.get("finished_at") or metadata.get("updated_at")
    if start and end:
        try:
            return max(
                0.0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
            )
        except (TypeError, ValueError):
            pass
    return None


def build_summary(cfg, metrics, stages, job_metadata):
    """Summarize measured outcomes without using test scores to tune anything."""
    values = metrics.get("metrics", {})
    for name in MODEL_NAMES:
        value = values.get(name, {}).get("auc")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Missing or invalid completed test AUC for {name}")
    if set(stages) != set(STAGE_NAMES):
        raise ValueError("Final delivery requires MF, Stage 1, and Stage 2 summaries")
    partial = []
    for stage, summary in stages.items():
        status = summary.get("status")
        if status not in {"complete", "time_limit"}:
            raise ValueError(f"Cannot finalize {stage} with status {status!r}")
        short = (
            stage in {"stage1", "stage2"}
            and summary.get("steps_completed", 0) < cfg[stage]["max_steps"]
        )
        if status == "time_limit" or short:
            partial.append(stage)
    collm = values["CoLLM-MF"]["auc"]
    deltas = {name: collm - values[name]["auc"] for name in ("MF", "TALLRec")}
    commit = job_metadata.get("git_commit")
    if not commit and isinstance(job_metadata.get("git"), dict):
        commit = job_metadata["git"].get("commit")
    reused = job_metadata.get("reused_stages")
    if reused is None:
        reused = [
            stage for stage, value in job_metadata.get("stages", {}).items() if value.get("reused")
        ]
    return {
        "run_kind": cfg["run_kind"],
        "completion_status": "time_limit" if partial else "complete",
        "partial_stages": partial,
        "model": cfg["model"]["id"],
        "model_revision": cfg["model"]["revision"],
        "git_commit": commit,
        "device": cfg["device"],
        "data_manifest": metrics["data_manifest"],
        "pipeline_elapsed_seconds": _pipeline_seconds(job_metadata),
        "training_elapsed_seconds": sum(
            float(stage.get("elapsed_seconds", 0)) for stage in stages.values()
        ),
        "current_training_elapsed_seconds": sum(
            float(value.get("elapsed_seconds", 0))
            for stage, value in stages.items()
            if stage not in reused
        ),
        "evaluation_elapsed_seconds": float(metrics.get("elapsed_seconds", 0)),
        "stages": {
            name: {
                "status": stage["status"],
                "steps_completed": stage.get("steps_completed", 0),
                "elapsed_seconds": stage.get("elapsed_seconds", 0),
                "best_valid_auc": stage.get("best_valid_auc"),
            }
            for name, stage in stages.items()
        },
        "metrics": values,
        "collm_auc_delta": deltas,
        "test_used_for_selection": False,
        "evaluation_protocol": metrics["evaluation_protocol"],
        "contract_sha256": metrics.get("contract_sha256"),
        "reused_stages": reused,
    }


def _outcome(delta, baseline):
    if delta > 0:
        return f"CoLLM-MF มี AUC สูงกว่า {baseline} {delta:.4f} ในการทดลองนี้"
    if delta < 0:
        return f"CoLLM-MF มี AUC ต่ำกว่า {baseline} {abs(delta):.4f} ในการทดลองนี้"
    return f"CoLLM-MF มี AUC เท่ากับ {baseline} ในการทดลองนี้"


def thai_summary(summary):
    smoke = summary["run_kind"] == "smoke"
    title = "ผลตรวจ pipeline เท่านั้น — ไม่ใช่ผลวิจัย" if smoke else "ผลการทดลอง CoLLM-MF"
    lines = [f"# {title}", ""]
    if smoke:
        lines += [
            "**ข้อมูลเล็กและ steps จำกัด ใช้ยืนยันการทำงานของโปรแกรม ห้ามใช้สรุปประสิทธิภาพสำหรับส่งงาน**",
            "",
        ]
    if summary["partial_stages"]:
        lines += [
            "**มีขั้นตอนที่หมดเวลาหรือทำไม่ครบจำนวน steps ตามแผน ผลนี้มาจาก checkpoint ที่ผ่าน validation แล้วเท่านั้น**",
            "",
        ]
    counts = summary["data_manifest"]
    lines += [
        f"- โมเดลฐาน: `{summary['model']}`",
        f"- Revision: `{summary['model_revision']}`",
        f"- Git commit: `{summary['git_commit'] or 'ไม่ได้บันทึก (smoke/test)'}`",
        f"- Device: `{summary['device']}`",
        f"- ข้อมูล train / validation / test: {counts['train']['count']:,} / {counts['valid']['count']:,} / {counts['test']['count']:,}",
        "",
        "| โมเดล | Test AUC |",
        "|---|---:|",
    ]
    lines += [f"| {name} | {summary['metrics'][name]['auc']:.4f} |" for name in MODEL_NAMES]
    lines += [""] + [
        _outcome(delta, baseline) for baseline, delta in summary["collm_auc_delta"].items()
    ]
    lines += [
        "",
        "ตัวเลขเป็นผลของการทดลองนี้ ไม่ใช่ข้อพิสูจน์ว่าจะดีกว่าบนทุก dataset และไม่ใช่การทำซ้ำตารางในเปเปอร์แบบตรงทุกค่า",
        "",
        "## เวลาที่ใช้และจำนวน steps",
        "",
        "| ขั้นตอน | สถานะ | Optimizer steps | เวลา HH:MM:SS |",
        "|---|---|---:|---:|",
    ]
    for stage, values in summary["stages"].items():
        lines.append(
            f"| {STAGE_NAMES[stage]} | {values['status']} | {values['steps_completed']:,} | {_duration(values['elapsed_seconds'])} |"
        )
    total = summary["pipeline_elapsed_seconds"]
    lines += [
        "",
        f"เวลาฝึกสะสมของ checkpoints ทั้งสามส่วน: {_duration(summary['training_elapsed_seconds'])}",
        f"เวลาประเมินผล: {_duration(summary['evaluation_elapsed_seconds'])}",
    ]
    if summary["reused_stages"]:
        lines += [
            f"ขั้นตอนที่ใช้ checkpoint จากรอบก่อน: {', '.join(summary['reused_stages'])}",
            f"เวลาฝึกที่เกิดในรอบนี้: {_duration(summary['current_training_elapsed_seconds'])}",
        ]
    if total is not None:
        lines.append(f"เวลารวม pipeline ก่อนแพ็กรายงาน: {_duration(total)}")
    lines += [
        "",
        "## วิธีอ่านผล",
        "",
        "- เลือก checkpoint ด้วย validation AUC; ไม่ใช้ test AUC เลือกค่าฝึก",
        "- AUC วัดบน ratings ที่สังเกตจริงใน test ไม่ใช่ full-catalog top-k recommendation",
        "- History ใช้ positive feedback ที่เกิดก่อนตัวอย่างเป้าหมาย; feedback ที่เกิดก่อนหน้าใน held-out period อาจเป็น history ของเหตุการณ์ถัดไป",
        "- ranking_example.csv เป็นตัวอย่างจัดอันดับ candidates ภายใต้ history เดียวกัน ไม่ใช่ metric เพิ่มเติม",
        "- Backbone ขนาดเล็ก, subset, ความยาว prompt และ training budget เป็นการปรับตามทรัพยากร ต้องระบุในการนำเสนอ",
        "",
        "รายละเอียดและค่าที่ใช้จริงอยู่ใน config_used.yaml, metrics.json, pipeline_summary.json และ REPORT.md",
        "",
    ]
    return "\n".join(lines)


def training_curves(output, results):
    histories = {}
    for stage in ("stage1", "stage2"):
        path = output / stage / "training.jsonl"
        if path.exists():
            histories[stage] = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    if not any(histories.values()):
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = {"stage1": "#2563eb", "stage2": "#0d9488"}
    has_validation = False
    for stage, rows in histories.items():
        if not rows:
            continue
        axes[0].plot(
            [row["step"] for row in rows],
            [row["train_loss"] for row in rows],
            label=STAGE_NAMES[stage],
            color=colors[stage],
        )
        measured = [row for row in rows if "valid_auc" in row]
        if measured:
            has_validation = True
            axes[1].plot(
                [row["step"] for row in measured],
                [row["valid_auc"] for row in measured],
                marker="o",
                label=STAGE_NAMES[stage],
                color=colors[stage],
            )
    axes[0].set(title="Training loss", xlabel="Optimizer step", ylabel="BCE loss")
    axes[1].set(title="Validation AUC", xlabel="Optimizer step", ylabel="ROC AUC", ylim=(0, 1))
    axes[0].legend()
    if has_validation:
        axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    target = results / "training_curves.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def _embedded_image(path, alt):
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{encoded}">'


def html_report(summary, results, archive_name):
    smoke = summary["run_kind"] == "smoke"
    title = "Smoke test — ไม่ใช่ผลวิจัย" if smoke else "ผลการทดลอง CoLLM-MF"
    notice = (
        "ตรวจ pipeline ด้วยข้อมูลเล็กเท่านั้น ห้ามใช้คะแนนนี้เป็นข้อสรุปผลวิจัย"
        if smoke
        else "ผลจากโมเดลและข้อมูลชุดที่ระบุด้านล่าง"
    )
    if summary["partial_stages"]:
        notice += " · บางขั้นตอนทำไม่ครบแผน ใช้เฉพาะ checkpoint ที่ผ่าน validation"
    rows = "".join(
        f"<tr><td>{name}</td><td>{summary['metrics'][name]['auc']:.4f}</td></tr>"
        for name in MODEL_NAMES
    )
    stages = "".join(
        f"<tr><td>{STAGE_NAMES[name]}</td><td>{html.escape(values['status'])}</td><td>{values['steps_completed']:,}</td><td>{_duration(values['elapsed_seconds'])}</td></tr>"
        for name, values in summary["stages"].items()
    )
    outcomes = "".join(
        f"<p>{_outcome(delta, name)}</p>" for name, delta in summary["collm_auc_delta"].items()
    )
    counts = summary["data_manifest"]
    runtime = (
        _duration(summary["pipeline_elapsed_seconds"])
        if summary["pipeline_elapsed_seconds"] is not None
        else "ดู pipeline.log"
    )
    images = _embedded_image(results / "auc_comparison.png", "AUC comparison") + _embedded_image(
        results / "training_curves.png", "Training and validation curves"
    )
    return f'''<!doctype html>
<html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f1f5f9;color:#0f172a;font:16px/1.65 "Leelawadee UI",Tahoma,sans-serif}}
main{{max-width:1000px;margin:32px auto;padding:32px;background:white;border-radius:16px}}h1{{margin:0 0 8px;font-size:30px}}h2{{margin-top:30px;font-size:21px}}.notice{{padding:12px 16px;background:#fff7ed;border-left:4px solid #f59e0b;border-radius:6px}}.meta{{color:#475569;overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%;margin:12px 0}}td,th{{padding:10px 14px;border-bottom:1px solid #e2e8f0;text-align:left}}th{{background:#f8fafc}}img{{max-width:100%;height:auto;margin:12px 0}}a{{color:#0369a1}}.links{{display:flex;gap:16px;flex-wrap:wrap}}code{{font-size:14px}}@media(max-width:650px){{main{{margin:0;padding:20px;border-radius:0}}h1{{font-size:25px}}}}
</style><main><h1>{title}</h1><p class="notice">{notice}</p>
<p class="meta">{html.escape(summary["model"])} · {html.escape(summary["device"])}<br>
Train / validation / test: {counts["train"]["count"]:,} / {counts["valid"]["count"]:,} / {counts["test"]["count"]:,}<br>
Revision: <code>{html.escape(summary["model_revision"])}</code><br>
Git commit: <code>{html.escape(summary["git_commit"] or "smoke/test — ไม่ได้บันทึก")}</code><br>เวลารวมก่อนแพ็กรายงาน: {runtime}</p>
<h2>ผลที่วัดได้</h2><table><thead><tr><th>โมเดล</th><th>Test AUC</th></tr></thead><tbody>{rows}</tbody></table>{outcomes}
<p>ผลนี้ไม่รับรองว่าจะดีกว่าบนทุก dataset และไม่ได้ใช้ test scores ปรับค่าฝึก</p>{images}
<h2>การฝึกที่เกิดขึ้นจริง</h2><table><thead><tr><th>ขั้นตอน</th><th>สถานะ</th><th>Optimizer steps</th><th>HH:MM:SS</th></tr></thead><tbody>{stages}</tbody></table>
<h2>ขอบเขตการประเมิน</h2><p>AUC วัดบน observed held-out ratings ไม่ใช่ full-catalog top-k recommendation เลือก checkpoint ด้วย validation เท่านั้น ประวัติใช้ positive feedback ที่เกิดก่อนเป้าหมายตาม rolling-history protocol และใช้โมเดลขนาดเล็ก/ข้อมูล/งบฝึกตาม config ที่บันทึกไว้</p>
<h2>ไฟล์ผลลัพธ์</h2><div class="links"><a href="SUMMARY_TH.md">สรุปภาษาไทย</a><a href="REPORT.md">รายงานละเอียด</a><a href="metrics.json">Metrics</a><a href="predictions.csv">Predictions CSV</a><a href="ranking_example.csv">ตัวอย่าง ranking</a><a href="config_used.yaml">Config ที่ใช้</a><a href="{html.escape(archive_name)}">Source + results ZIP</a></div>
</main></html>'''


def finalize_run(cfg, job_dir, job_metadata):
    """Create all human-readable deliverables after complete test-set evaluation."""
    results = results_path(cfg)
    output = project_path(cfg["output_dir"], "outputs")
    metrics = read_json(results / "metrics.json")
    stages = {stage: read_json(output / stage / "summary.json") for stage in STAGE_NAMES}
    summary = build_summary(cfg, metrics, stages, job_metadata)
    job_dir = (ROOT / job_dir).resolve()
    summary["job_dir"] = str(job_dir.relative_to(ROOT))
    run_name = output.name
    archive = results / f"CoLLM_{run_name}_source_results.zip"
    write_json(results / "pipeline_summary.json", summary)
    (results / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (results / "SUMMARY_TH.md").write_text(thai_summary(summary), encoding="utf-8")
    for stage in STAGE_NAMES:
        shutil.copyfile(output / stage / "summary.json", results / f"{stage}_summary.json")
        log = output / stage / "training.jsonl"
        if log.exists():
            shutil.copyfile(log, results / f"{stage}_training.jsonl")
    if (job_dir / "pipeline.log").exists():
        shutil.copyfile(job_dir / "pipeline.log", results / "pipeline.log")
    training_curves(output, results)
    (results / "report.html").write_text(
        html_report(summary, results, archive.name), encoding="utf-8"
    )
    package_project(ROOT, archive, results)
    return {
        "results_dir": results.relative_to(ROOT).as_posix(),
        "source_zip": archive.relative_to(ROOT).as_posix(),
        "completion_status": summary["completion_status"],
    }
