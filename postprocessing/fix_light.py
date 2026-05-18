import os
from pathlib import Path

import cv2
import numpy as np
import argparse
import sys
import json
import mediapipe as mp
import subprocess
import torch
from models.shadow_remover import ShadowRemovalNet
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from rembg import remove
from masking.mediapipe_mask import FaceMaskGenerator

_ROOT = Path(__file__).resolve().parents[1]


def ensure_model(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(["python", "download_models.py", path])


def parse_args():
    parser = argparse.ArgumentParser(description="ICAO Image Processing Pipeline")
    parser.add_argument("--input", required=True, help="Image input")
    parser.add_argument("--mask", default=None, help="Chemin vers un masque (image)")
    parser.add_argument("--landmarks", default=None, help="Chemin vers les landmarks (JSON)")
    parser.add_argument("--fix-bg", action="store_true", help="Fix background")
    parser.add_argument("--fix-light", action="store_true", help="Fix directional light")
    parser.add_argument("--fix-icao", action="store_true", help="Fix ICAO cadre")
    parser.add_argument("--fix-blurred", action="store_true", help="Fix blurred image")
    parser.add_argument("--output", default=None, help="Chemin de sauvegarde du résultat")
    parser.add_argument("--show", action="store_true", help="Afficher l'image résultat")
    return parser.parse_args()


FACE_MODEL = str(_ROOT / "models" / "deploy.prototxt")
FACE_WEIGHTS = str(_ROOT / "models" / "res10_300x300_ssd_iter_140000.caffemodel")
LANDMARK_MODEL = str(_ROOT / "models" / "lbfmodel.yaml")
SHADOW_MODEL_PATH = str(_ROOT / "checkpoints" / "shadow_removal_best.pth")
SHADOW_INPUT_SIZE = 512
_shadow_model = None

def init_landmarks():
    ensure_model(FACE_MODEL)
    ensure_model(FACE_WEIGHTS)
    ensure_model(LANDMARK_MODEL)
    net = cv2.dnn.readNetFromCaffe(FACE_MODEL, FACE_WEIGHTS)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(LANDMARK_MODEL)
    return net, facemark


def get_landmarks(image, net, facemark):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            face_rect = np.array([[x1, y1, x2 - x1, y2 - y1]])
            ok, landmarks = facemark.fit(image, face_rect)
            if ok:
                return landmarks[0][0]
    return None


#  SEGMENTATION
BASE_OPTIONS = mp_python.BaseOptions(
    model_asset_path="models/selfie_segmentation.tflite"
)
SEGMENTER_OPTIONS = vision.ImageSegmenterOptions(
    base_options=BASE_OPTIONS,
    output_category_mask=True,
    running_mode=vision.RunningMode.IMAGE,
)
def get_segmentation_mask(image):
    ensure_model("models/selfie_segmentation.tflite")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with vision.ImageSegmenter.create_from_options(SEGMENTER_OPTIONS) as segmenter:
        result = segmenter.segment(mp_image)
        mask = result.category_mask.numpy_view().astype(np.float32)
        return mask / 255.0

def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Impossible de charger le masque : {path}")
    return (mask / 255.0).astype(np.float32)


def load_landmarks(path):
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data, dtype=np.float32)


def compute_initial_data(image, net, facemark, mask_path=None, landmarks_path=None):
    """Calcule ou charger masque et landmarks."""
    if mask_path:
        mask = load_mask(mask_path)
    else:
        mask = get_segmentation_mask(image)

    if landmarks_path:
        landmarks = load_landmarks(landmarks_path)
    else:
        landmarks = get_landmarks(image, net, facemark)

    return mask, landmarks


# - MASQUES DÉRIVÉS -

def smooth_mask(mask):
    return cv2.GaussianBlur(mask, (31, 31), 0)


def get_face_bbox_from_mask(mask):
    binary = (mask > 0.5).astype(np.uint8)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None, None

    x, y, w, h = cv2.boundingRect(coords)
    mask_tete = binary[y:y + h // 4, :]
    coords_tete = cv2.findNonZero(mask_tete)
    if coords_tete is None:
        return None, None

    _, _, w2, _ = cv2.boundingRect(coords_tete)
    return y, w2

#  MASK REMBG (pour fix_background uniquement)

def get_mask_rembg(image):
    alpha = remove(image)[:, :, 3]
    mask = (alpha > 128).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return (mask / 255.0).astype(np.float32)


def fix_background(image):
    h, w = image.shape[:2]
    cs = max(20, h // 10)

    mask = get_mask_rembg(image)

    # Eroder le masque pour couper les bords douteux
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_clean = cv2.erode(mask, erode_k)
    mask_clean = cv2.GaussianBlur(mask_clean, (5, 5), 0)

    # Composition fond blanc directement avec masque erodé
    mask_3 = mask_clean[:, :, np.newaxis]
    result = image.astype(np.float32) * mask_3 + np.full(image.shape, 240, np.float32) * (1 - mask_3)
    return result.clip(0, 255).astype(np.uint8)

# FIX LIGHT

def load_shadow_model():
    global _shadow_model
    if _shadow_model is not None:
        return _shadow_model

    if not os.path.exists(SHADOW_MODEL_PATH):
        # On lève une exception proprement au lieu de tuer le programme
        raise FileNotFoundError(f"Erreur: modèle shadow introuvable à {SHADOW_MODEL_PATH}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = ShadowRemovalNet()
    state = torch.load(SHADOW_MODEL_PATH, map_location=device)

    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device).eval()
    _shadow_model = (model, device)
    return _shadow_model

_face_mask_gen = None

def get_face_mask_gen():
    global _face_mask_gen
    if _face_mask_gen is None:
        _face_mask_gen = FaceMaskGenerator(SHADOW_INPUT_SIZE)
    return _face_mask_gen


# MediaPipe jawline landmark indices (bottom of face oval, left→chin→right).
# These trace the exact chin curve for any face shape, rotation or size.
_JAWLINE = [
    132, 58, 172, 136, 150, 149, 176, 148, 152,   # left jaw → chin
    377, 400, 378, 379, 365, 397, 288, 361,         # chin → right jaw
]


def _build_neck_mask_smart(face_pts: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Build a precise neck mask from actual MediaPipe jawline landmarks.

    Top boundary  = real jaw curve (follows chin shape, tilt, asymmetry).
    Bottom edge   = jaw curve translated downward by 40 % of face height.
    Result        = closed polygon filled, then soft-blurred for blending.

    Much more accurate than a fixed rectangle: works for tilted heads,
    short/long necks, beards, wide/narrow jaws.
    """
    if face_pts is None or len(face_pts) < 478:
        return np.zeros((h, w), dtype=np.float32)

    face_h = float(face_pts[:, 1].max() - face_pts[:, 1].min())
    if face_h < 10:
        return np.zeros((h, w), dtype=np.float32)

    # Extract jawline points and sort left→right for a clean polygon edge
    jaw_pts = face_pts[_JAWLINE].copy()
    jaw_pts = jaw_pts[jaw_pts[:, 0].argsort()]

    # Neck depth proportional to face height
    neck_depth = int(face_h * 0.42)

    # Top edge  = actual jawline curve
    top_edge    = jaw_pts.astype(np.float32)
    # Bottom edge = same curve pushed straight down
    bottom_edge = top_edge.copy()
    bottom_edge[:, 1] = np.clip(bottom_edge[:, 1] + neck_depth, 0, h - 1)

    # Closed polygon: top L→R then bottom R→L
    poly = np.vstack([top_edge, bottom_edge[::-1]]).astype(np.int32)

    neck_mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(neck_mask, [poly], 1.0)

    # Zero out the face zone (above the highest jaw point) so no overlap with face mask
    chin_y = int(jaw_pts[:, 1].min())
    neck_mask[:chin_y, :] = 0.0

    # Apply vertical gradient: strongest at chin, fades to zero at bottom
    neck_binary = (neck_mask > 0).astype(np.float32)
    ys = np.arange(h, dtype=np.float32)
    chin_line   = float(jaw_pts[:, 1].max())          # actual chin bottom
    neck_bot    = chin_line + neck_depth
    grad        = np.clip(1.0 - (ys - chin_line) / max(1.0, neck_bot - chin_line), 0.0, 1.0)
    neck_mask   = neck_mask * grad[:, None]

    # Smooth edges for invisible blending
    blur_k = max(15, int(face_h * 0.07) | 1)
    neck_mask = cv2.GaussianBlur(neck_mask, (blur_k, blur_k), 0)
    return np.clip(neck_mask, 0.0, 1.0)


def fix_light(image):
    """
    Face shadow removal with neck extension.
    - Face zone  : full model prediction + texture preservation
    - Neck zone  : luminance-only transfer (LAB L-channel) to avoid color
                   artifacts from the model never being trained on neck pixels
    - Gradient mask ensures smooth chin→neck transition
    """
    model, device = load_shadow_model()
    orig_h, orig_w = image.shape[:2]

    # 1. Basse résolution pour l'inférence
    resized = cv2.resize(image, (SHADOW_INPUT_SIZE, SHADOW_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb_lr = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 2. Détection visage + landmarks en un seul appel MediaPipe
    mask_gen = get_face_mask_gen()
    face_mask_lr, face_pts_lr = mask_gen.detect(rgb_lr)
    face_mask_lr = face_mask_lr.astype(np.float32)

    # Translation vers le haut pour couvrir le front
    shift_y = -20
    M = np.float32([[1, 0, 0], [0, 1, shift_y]])
    shifted_mask = cv2.warpAffine(face_mask_lr, M, (SHADOW_INPUT_SIZE, SHADOW_INPUT_SIZE))
    face_mask_lr = cv2.max(face_mask_lr, shifted_mask)
    face_mask_lr = cv2.GaussianBlur(face_mask_lr, (31, 31), 0)
    face_mask_lr = np.clip(face_mask_lr, 0.0, 1.0)
    mask_bool_lr = face_mask_lr > 0.5

    # Masque cou précis — polygone construit depuis les landmarks du menton réel
    neck_mask_lr = _build_neck_mask_smart(face_pts_lr, SHADOW_INPUT_SIZE, SHADOW_INPUT_SIZE)

    # 3. Inférence ShadowRemoval
    tensor = (torch.from_numpy(rgb_lr.astype(np.float32) / 255.0)
              .permute(2, 0, 1).unsqueeze(0).to(device))
    with torch.no_grad():
        pred = model(tensor)
    pred_rgb_lr = np.clip(pred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    pred_bgr_lr = cv2.cvtColor(pred_rgb_lr, cv2.COLOR_RGB2BGR)

    # 4. Garde adaptatif
    gray_lr = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    std_dev  = float(np.std(gray_lr[mask_bool_lr])) if np.any(mask_bool_lr) else 50.0
    diff     = cv2.absdiff(resized, pred_bgr_lr)
    mean_diff = float(diff.mean(axis=2)[mask_bool_lr].mean()) if np.any(mask_bool_lr) else 0.0

    if mean_diff < (std_dev * 0.15):
        return image

    # 5. Force adaptative
    dynamic_strength = float(np.clip(0.40 + (mean_diff / 100.0), 0.45, 0.60))

    # 6. Upscale prédiction + masques
    pred_bgr_hr  = cv2.resize(pred_bgr_lr,  (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    face_mask_hr = cv2.resize(face_mask_lr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    neck_mask_hr = cv2.resize(neck_mask_lr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Érosion légère du masque visage
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    face_mask_hr = cv2.erode(face_mask_hr, inner_kernel)
    face_mask_hr = np.clip(face_mask_hr, 0.0, 1.0)

    # 7. Texture HF
    image_f   = image.astype(np.float32)
    pred_f    = pred_bgr_hr.astype(np.float32)
    blur_orig = cv2.GaussianBlur(image_f, (0, 0), sigmaX=2.0)
    texture_hf = np.clip(image_f - blur_orig, -15.0, 15.0)

    pred_soft     = cv2.GaussianBlur(pred_f, (0, 0), sigmaX=1.5)
    pred_textured = pred_soft + texture_hf

    # 8. Fusion visage
    alpha_face = (cv2.GaussianBlur(face_mask_hr, (15, 15), 0) * dynamic_strength)[..., None]
    blended    = image_f * (1.0 - alpha_face) + pred_textured * alpha_face

    # 9. Correction cou — colonne par colonne
    # Un seul scalaire moyen ne fonctionne pas sur les visages éclairés de façon
    # directionnelle : le côté clair annule le côté sombre → delta ≈ 0.
    # Solution : pour chaque colonne x, mesurer la correction L au bas du visage
    # (menton) et l'extrapoler à la colonne correspondante du cou.
    # Cela suit naturellement la direction de l'ombre.
    orig_lab_lr = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    pred_lab_lr = cv2.cvtColor(pred_bgr_lr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_delta_lr  = pred_lab_lr[:, :, 0] - orig_lab_lr[:, :, 0]  # (H, W)

    face_hard = (face_mask_lr > 0.5).astype(np.uint8)
    coords_h  = cv2.findNonZero(face_hard)

    if coords_h is not None:
        bx_h, by_h, bw_h, bh_h = cv2.boundingRect(coords_h)

        # Strip menton = bas 20 % de la bbox visage
        chin_top = by_h + int(bh_h * 0.80)
        chin_bot = by_h + bh_h
        chin_strip = np.zeros(face_mask_lr.shape, dtype=np.float32)
        chin_strip[chin_top:chin_bot, bx_h:bx_h + bw_h] = 1.0
        chin_strip *= (face_mask_lr > 0.4).astype(np.float32)

        # Moyenne L-delta par colonne (vectorisée, pas de boucle Python)
        col_sum   = (l_delta_lr * chin_strip).sum(axis=0)       # (W,)
        col_count = chin_strip.sum(axis=0)                       # (W,)
        l_corr_col = np.where(col_count > 0,
                              col_sum / (col_count + 1e-6),
                              0.0).astype(np.float32)            # (W,)

        # Lissage horizontal pour éviter les artéfacts entre colonnes
        l_corr_col = cv2.GaussianBlur(
            l_corr_col.reshape(1, -1), (31, 1), 0
        ).reshape(-1)

        # Carte 2D : chaque ligne du cou hérite de la correction de sa colonne
        l_corr_map_lr = np.tile(l_corr_col, (SHADOW_INPUT_SIZE, 1))  # (H, W)

        # Upscale à la résolution originale
        l_corr_map_hr = cv2.resize(
            l_corr_map_lr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )

        # N'appliquer que si la correction est significative (évite les photos déjà bien éclairées)
        max_corr = float(np.abs(l_corr_col[bx_h:bx_h + bw_h]).max()) if bw_h > 0 else 0.0
        if max_corr > 0.5:
            orig_lab_hr = cv2.cvtColor(
                np.clip(image_f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB
            ).astype(np.float32)

            neck_corrected_lab          = orig_lab_hr.copy()
            neck_corrected_lab[:, :, 0] = np.clip(
                orig_lab_hr[:, :, 0] + l_corr_map_hr, 0, 255
            )
            neck_corrected_bgr = cv2.cvtColor(
                neck_corrected_lab.astype(np.uint8), cv2.COLOR_LAB2BGR
            ).astype(np.float32)

            alpha_neck = (neck_mask_hr * dynamic_strength)[..., None]
            blended    = blended * (1.0 - alpha_neck) + neck_corrected_bgr * alpha_neck

    return np.clip(blended, 0, 255).astype(np.uint8)

def fix_icao(image, landmarks, mask):
    """
    Recadre au format ICAO  (35x45mm).
    Mesure la tête réelle (sommet→menton) et calcule le cadre
    pour que la tête fasse 34mm sur 45mm avec 4mm de marge.
    """
    ECHEC = (image, 0, 0, 1.0, 1.0)
    hauteur, largeur = image.shape[:2]

    if landmarks is None:
        return ECHEC

    points = landmarks
    centre_yeux = (points[36] + points[45]) / 2.0
    centre_visage_x = (points[0][0] + points[16][0]) / 2.0
    menton = points[8]
    face_width_landmarks = np.linalg.norm(points[16] - points[0])
    haut_visage = np.linalg.norm(menton - centre_yeux)

    if haut_visage < 50:
        return ECHEC

    # Sommet de tête : fallback landmarks
    brow_top = min(points[19][1], points[24][1])
    larg_ref = max(
        face_width_landmarks * 1.05,
        np.linalg.norm(points[15] - points[1]) * 1.1
    )
    facteur_crane = np.clip(0.75 + 0.15 * (larg_ref / haut_visage), 0.75, 1.1)
    sommet_tete = max(0, brow_top - facteur_crane * haut_visage)

    # Masque améliore si fiable
    sommet_mask, largeur_mask = get_face_bbox_from_mask(mask)
    if sommet_mask is not None:
        if largeur_mask * 1.02 < face_width_landmarks * 1.8:
            sommet_tete = max(0, sommet_mask)

    # Marge accessoires
    sommet_tete = max(0, sommet_tete - haut_visage * 0.08)

    hauteur_tete = menton[1] - sommet_tete
    if hauteur_tete < 50:
        return ECHEC

    # Dimensions du cadre depuis la norme ICAO
    RATIO_TETE = 34.0 / 45.0
    RATIO_MARGE_HAUT = 4.0 / 45.0
    RATIO_CIBLE = 35.0 / 45.0

    recad_haut = int(hauteur_tete / RATIO_TETE)
    recad_larg = int(recad_haut * RATIO_CIBLE)

    if recad_larg < 200 or recad_haut < 250:
        return ECHEC

    # Positionnement
    y1 = int(sommet_tete - RATIO_MARGE_HAUT * recad_haut)
    y2 = y1 + recad_haut
    x1 = int(centre_visage_x - recad_larg / 2)
    x2 = x1 + recad_larg

    # Validation yeux dans zone ICAO
    pos_yeux_ratio = (centre_yeux[1] - y1) / recad_haut
    if not (0.40 <= pos_yeux_ratio <= 0.70):
        y1 = int(centre_yeux[1] - 0.55 * recad_haut)
        y2 = y1 + recad_haut

    # Clamp aux bords
    if y1 < 0:       y2 -= y1;            y1 = 0
    if y2 > hauteur: y1 -= y2 - hauteur;  y2 = hauteur; y1 = max(0, y1)
    if x1 < 0:       x2 -= x1;            x1 = 0
    if x2 > largeur: x1 -= x2 - largeur;  x2 = largeur; x1 = max(0, x1)

    if x2 <= x1 or y2 <= y1:
        return ECHEC
    # Vérification menton
    if menton[1] > y2 - 10:
        return ECHEC

    # Crop + ajustement ratio
    crop = image[y1:y2, x1:x2]
    h_c, w_c = crop.shape[:2]

    if h_c < 10 or w_c < 10:
        return ECHEC

    r = w_c / h_c
    if r > RATIO_CIBLE:
        nw = int(h_c * RATIO_CIBLE)
        dx = (w_c - nw) // 2
        crop = crop[:, dx:dx + nw]
        x1 += dx
    elif r < RATIO_CIBLE:
        nh = int(w_c / RATIO_CIBLE)
        dy = (h_c - nh) // 2
        crop = crop[dy:dy + nh, :]
        y1 += dy

    # Redimensionnement final
    out_h = int(np.sqrt(largeur * hauteur / RATIO_CIBLE))
    out_w = int(out_h * RATIO_CIBLE)

    scale_x = out_w / crop.shape[1]
    scale_y = out_h / crop.shape[0]

    resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    return resized, x1, y1, scale_x, scale_y

def fix_blurred(image):
    """
    Deblur intelligent (version PRO ICAO)

    ✔ Détection fiable du flou
    ✔ Sharpen uniquement si nécessaire
    ✔ Pas d'amplification du bruit
    ✔ Préserve texture naturelle
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Détection du flou (plus fiable)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # seuils réalistes
    if variance > 120:
        # image déjà nette → on ne touche pas
        return image

    if variance > 80:
        strength = 0.3
    elif variance > 40:
        strength = 0.6
    else:
        strength = 1.0

    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)

    sharpen = cv2.addWeighted(
        image, 1 + strength,
        blur, -strength,
        0
    )

    result = cv2.bilateralFilter(sharpen, 5, 20, 20)

    return result


#  MAIN

def main():
    args = parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print("Erreur image", file=sys.stderr)
        sys.exit(1)

    net, facemark = None, None
    mask, landmarks = None, None

    if args.fix_icao:
        net, facemark = init_landmarks()
        mask, landmarks = compute_initial_data(
            img, net, facemark,
            mask_path=args.mask,
            landmarks_path=args.landmarks
        )

    result = img.copy()

    if not any([args.fix_icao, args.fix_bg, args.fix_light, args.fix_blurred]):
        print("Aucun traitement appliqué")

    # 1. FIX BG D'ABORD : Détourage sur l'image entière
    if args.fix_bg:
        result = fix_background(result)

    # 2.  Recadrage de l'image déjà détourée

    if args.fix_icao:
        result, cx, cy, sx, sy = fix_icao(result, landmarks, mask)

        if cx == 0 and cy == 0 and sx == 1.0 and sy == 1.0:
            print("  fix_icao: recadrage échoué, image originale conservée")
        else:
            h_new, w_new = result.shape[:2]
            crop_h, crop_w = int(h_new / sy), int(w_new / sx)
            mask = cv2.resize(mask[cy:cy + crop_h, cx:cx + crop_w],
                              (w_new, h_new), interpolation=cv2.INTER_LINEAR)
            if landmarks is not None:
                landmarks[:, 0] = (landmarks[:, 0] - cx) * sx
                landmarks[:, 1] = (landmarks[:, 1] - cy) * sy

    # 3. FIX LIGHT : Correction lumière sur l'image recadrée
    if args.fix_light:
        result = fix_light(result)

    # 4. FIX BLURRED : Netteté à la fin

    if args.fix_blurred:
        result = fix_blurred(result)

    if args.output:
        success = cv2.imwrite(args.output, result)
        if not success:
            print(f"Erreur: échec d'écriture vers {args.output}", file=sys.stderr)
            sys.exit(1)
        print(f"Sauvegardé : {args.output}")

    if args.show:
        cv2.imshow("original", img)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()