#!/usr/bin/env python
"""Single-image inference for the Dementia MRI research prototype.

    python predict.py --image /path/to/mri.jpg
    python predict.py --image mri.jpg --checkpoint checkpoints/best_model.pth --mc-dropout 20 --target-class "Mild Dementia"

Outputs a console report, prediction.json and Grad-CAM images. Not a diagnostic device.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.inference import DementiaPredictor, format_report, save_prediction_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Dementia MRI category prediction (research prototype)")
    ap.add_argument("--image", default=None, help="brain MRI slice (JPG/PNG)")
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best_model.pth"))
    ap.add_argument("--meta", default=None, help="inference_meta.json (temperature, thresholds); default: next to checkpoint")
    ap.add_argument("--output-dir", default=str(ROOT / "predictions"))
    ap.add_argument("--no-gradcam", action="store_true")
    ap.add_argument("--single-layer", action="store_true", help="standard (final-layer) Grad-CAM only")
    ap.add_argument("--target-class", default=None, help="explain this class instead of the predicted one")
    ap.add_argument("--mc-dropout", type=int, default=0, help="number of MC-dropout passes (0 = off)")
    a = ap.parse_args()

    image_path_str = a.image
    if not image_path_str:
        try:
            image_path_str = input("Enter path to brain MRI image: ").strip().strip('"\'')
        except (KeyboardInterrupt, EOFError):
            sys.exit("\nAborted.")
        if not image_path_str:
            sys.exit("Error: No image path provided.")

    img = Path(image_path_str)
    if not img.exists():
        sys.exit(f"Image not found: {img}")
    ckpt = Path(a.checkpoint)
    if not ckpt.exists():
        print(f"Default checkpoint not found at: {ckpt}")
        try:
            custom_ckpt = input("Enter checkpoint path (or press Enter to abort): ").strip().strip('"\'')
            if custom_ckpt:
                ckpt = Path(custom_ckpt)
        except (KeyboardInterrupt, EOFError):
            sys.exit("\nAborted.")
        if not ckpt.exists():
            sys.exit(f"Checkpoint not found: {ckpt} (train the model first or pass --checkpoint)")
    out = Path(a.output_dir) / img.stem
    out.mkdir(parents=True, exist_ok=True)
    p = DementiaPredictor(ckpt, a.meta)
    if a.target_class and a.target_class not in p.classes:
        sys.exit(f"--target-class must be one of {p.classes}")
    r = p.predict(img, gradcam_dir=None if a.no_gradcam else out, target_class=a.target_class,
                  mc_passes=a.mc_dropout, multi_layer=not a.single_layer)
    print(format_report(r))
    js = save_prediction_json(r, out / "prediction.json")
    print(f"\nprediction.json -> {js}")

    # Stage 2: Reflex Brain Tumor Check if Dementia is Normal (Non Demented)
    gradcam_file = r.get("gradcam_panel") or r.get("gradcam")
    if r["prediction"] == "Non Demented":
        print("\n" + "=" * 60)
        print("STAGE 2: REFLEX BRAIN TUMOR SCREENING (Triggered: Dementia Normal)")
        print("=" * 60)
        try:
            from src.tumor_detector import BrainTumorClassifier
            btc = BrainTumorClassifier.get_instance()
            t_res = btc.predict(img, out_dir=out)
            prob_pct = t_res["probability"] * 100.0
            print(f"Tumor Finding:     {t_res['label']}")
            print(f"Tumor Probability: {prob_pct:.2f}%")
            if t_res.get("layers_fused"):
                print(f"Layers Fused:      {', '.join(t_res['layers_fused'])}")
            if t_res.get("gradcam_panel"):
                print(f"Diagnostic Panel:  {t_res['gradcam_panel']}")
            if t_res["has_tumor"]:
                print("Consolidated:      ABNORMAL (Dementia Normal | BRAIN TUMOR DETECTED)")
                if t_res.get("gradcam_panel"):
                    gradcam_file = t_res["gradcam_panel"]
                elif t_res.get("gradcam_overlay"):
                    gradcam_file = t_res["gradcam_overlay"]
            else:
                print("Consolidated:      ALL CLEAR (No Dementia & No Brain Tumor Detected)")
        except Exception as te:
            print(f"Warning: Brain tumor screening could not run: {te}")
        print("=" * 60)
    else:
        print("\n[Protocol Note] Dementia detected: Stage 2 Brain Tumor screening bypassed.")

    # Automatically pop up the Grad-CAM visualization window/viewer
    if gradcam_file and Path(gradcam_file).exists():
        try:
            import os
            print(f"\nOpening visual output popup: {gradcam_file}")
            os.startfile(gradcam_file)
        except Exception as e:
            try:
                from PIL import Image
                Image.open(gradcam_file).show()
            except Exception:
                pass


if __name__ == "__main__":
    if len(sys.argv) == 1:
        from src.gui import main as run_gui
        run_gui()
    else:
        main()
