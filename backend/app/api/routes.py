"""
api/routes.py
─────────────
All FastAPI route definitions for the Hysteresis Loss Simulation API.
Endpoints are versioned under /api/v1/.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, File, status
from pydantic import BaseModel, Field

from app.utils.formulas import (
    steinmetz_core_loss,
    total_bertotti_loss,
    generate_hysteresis_loop,
    find_saturation_knee_point,
    calculate_loop_area,
    classify_operating_region
)
from app.utils.data_parser import parse_csv_to_bh_dataset, bh_dataset_to_arrays
from app.models.hysteresis_model import BHCurveAnalyser, SteinmetzModel

router = APIRouter(prefix="/api/v1", tags=["simulation"])


# ─── Pydantic request / response schemas ──────────────────────────────────────

class SteinmetzRequest(BaseModel):
    k:      float = Field(..., gt=0, description="Steinmetz coefficient k")
    f:      float = Field(..., gt=0, description="Frequency (Hz)")
    b_peak: float = Field(..., gt=0, description="Peak flux density (T)")
    alpha:  float = Field(1.7,       description="Frequency exponent α")
    beta:   float = Field(2.0,       description="Flux density exponent β")


class BertottiRequest(BaseModel):
    k_h:    float = Field(..., gt=0)
    k_e:    float = Field(..., gt=0)
    k_ex:   float = Field(..., gt=0)
    f:      float = Field(..., gt=0)
    b_peak: float = Field(..., gt=0)
    n:      float = Field(2.0)


class BHCurveRequest(BaseModel):
    h_values: list[float] = Field(..., min_length=3)
    b_values: list[float] = Field(..., min_length=3)
    material_name: str = Field("Unknown", max_length=80)


class HysteresisLoopRequest(BaseModel):
    h_max:        float = Field(..., gt=0)
    b_sat:        float = Field(..., gt=0)
    coercivity_h: float = Field(..., gt=0)
    remanence_b:  float = Field(..., gt=0)
    n_points:     int   = Field(200, ge=20, le=2000)
    frequency:    float = Field(0.0, ge=0)


class FitDataPoint(BaseModel):
    frequency: float = Field(..., gt=0)
    b_peak: float = Field(..., gt=0)
    p_loss: float = Field(..., gt=0)


class SteinmetzFitRequest(BaseModel):
    data_points: list[FitDataPoint] = Field(..., min_length=3)


# ─── Health check ─────────────────────────────────────────────────────────────

@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict:
    """Liveness probe. Returns 200 OK with service metadata."""
    return {
        "status": "ok",
        "service": "Hysteresis Loss Simulation API",
        "version": "1.0.0",
    }


# ─── Presets ──────────────────────────────────────────────────────────────────

@router.get("/presets", status_code=status.HTTP_200_OK)
async def get_presets() -> dict:
    """Get the physical presets of different materials for classroom simulation."""
    return {
        "silicon_steel_m4": {
            "name": "Electrical Silicon Steel (M4)",
            "description": "Transformer and motor lamination steel with low coercivity.",
            "b_sat": 1.80,
            "coercivity_h": 40.0,
            "remanence_b": 1.35,
            "h_max": 800.0,
            "core_volume_cm3": 500.0,
            "k": 0.015,
            "k_h": 0.012,
            "alpha": 1.7,
            "beta": 2.0,
            "k_e": 1.5e-5,
            "k_ex": 8e-5
        },
        "soft_pure_iron": {
            "name": "Soft Pure Iron",
            "description": "High saturation soft magnet for electromagnets and shielding.",
            "b_sat": 2.15,
            "coercivity_h": 80.0,
            "remanence_b": 1.15,
            "h_max": 1200.0,
            "core_volume_cm3": 250.0,
            "k": 0.025,
            "k_h": 0.020,
            "alpha": 1.65,
            "beta": 2.1,
            "k_e": 2.5e-5,
            "k_ex": 1.1e-4
        },
        "permalloy_78": {
            "name": "Permalloy (78% Ni, 22% Fe)",
            "description": "Very high permeability nickel-iron alloy for precision magnetic components.",
            "b_sat": 0.75,
            "coercivity_h": 1.6,
            "remanence_b": 0.55,
            "h_max": 15.0,
            "core_volume_cm3": 50.0,
            "k": 0.003,
            "k_h": 0.002,
            "alpha": 1.55,
            "beta": 2.05,
            "k_e": 5e-6,
            "k_ex": 1.5e-5
        },
        "mnzn_ferrite": {
            "name": "Soft MnZn Ferrite (High Freq)",
            "description": "High-resistivity soft ferrite for high-frequency inductors and transformers.",
            "b_sat": 0.40,
            "coercivity_h": 12.0,
            "remanence_b": 0.22,
            "h_max": 100.0,
            "core_volume_cm3": 10.0,
            "k": 0.0012,
            "k_h": 0.0008,
            "alpha": 1.6,
            "beta": 2.4,
            "k_e": 2e-7,
            "k_ex": 5e-6
        },
        "metglas_2605sa1": {
            "name": "Metglas 2605SA1 (Amorphous)",
            "description": "Amorphous ribbon with low coercivity and low magnetostriction.",
            "b_sat": 1.56,
            "coercivity_h": 4.0,
            "remanence_b": 0.85,
            "h_max": 80.0,
            "core_volume_cm3": 400.0,
            "k": 0.004,
            "k_h": 0.003,
            "alpha": 1.58,
            "beta": 1.95,
            "k_e": 4e-6,
            "k_ex": 2e-5
        },
        "hyperco_50": {
            "name": "Hyperco 50 (Cobalt-Iron)",
            "description": "High-saturation cobalt-iron alloy for force-dense magnetic actuators.",
            "b_sat": 2.40,
            "coercivity_h": 110.0,
            "remanence_b": 1.65,
            "h_max": 2000.0,
            "core_volume_cm3": 150.0,
            "k": 0.032,
            "k_h": 0.025,
            "alpha": 1.66,
            "beta": 2.05,
            "k_e": 3e-5,
            "k_ex": 1.3e-4
        },
        "supermalloy": {
            "name": "Supermalloy Premium",
            "description": "Ultra-high initial permeability nickel-iron alloy.",
            "b_sat": 0.79,
            "coercivity_h": 0.35,
            "remanence_b": 0.40,
            "h_max": 5.0,
            "core_volume_cm3": 120.0,
            "k": 0.0015,
            "k_h": 0.001,
            "alpha": 1.52,
            "beta": 2.0,
            "k_e": 3e-6,
            "k_ex": 1e-5
        },
        "alnico_v": {
            "name": "Alnico V (Hard Magnet)",
            "description": "Permanent magnet alloy with high remanence and high coercivity.",
            "b_sat": 1.25,
            "coercivity_h": 50000.0,
            "remanence_b": 1.10,
            "h_max": 150000.0,
            "core_volume_cm3": 30.0,
            "k": 0.18,
            "k_h": 0.15,
            "alpha": 1.5,
            "beta": 1.8,
            "k_e": 8e-5,
            "k_ex": 3e-4
        },
        "ceramic_hard_ferrite": {
            "name": "Ceramic Hard Ferrite",
            "description": "Permanent ceramic magnet with high coercivity and low conductivity.",
            "b_sat": 0.38,
            "coercivity_h": 170000.0,
            "remanence_b": 0.35,
            "h_max": 400000.0,
            "core_volume_cm3": 80.0,
            "k": 0.095,
            "k_h": 0.08,
            "alpha": 1.45,
            "beta": 1.75,
            "k_e": 5e-7,
            "k_ex": 2e-5
        },
        "ndfeb": {
            "name": "Neodymium NdFeB (Hard Supermagnet)",
            "description": "Rare-earth permanent magnet with very high coercivity.",
            "b_sat": 1.28,
            "coercivity_h": 840000.0,
            "remanence_b": 1.20,
            "h_max": 2000000.0,
            "core_volume_cm3": 20.0,
            "k": 0.24,
            "k_h": 0.20,
            "alpha": 1.45,
            "beta": 1.7,
            "k_e": 1.2e-4,
            "k_ex": 4e-4
        }
    }


# ─── Steinmetz core loss ──────────────────────────────────────────────────────

@router.post("/simulate/steinmetz")
async def simulate_steinmetz(req: SteinmetzRequest) -> dict:
    """
    Calculate core loss using the generalised Steinmetz equation.
    """
    try:
        loss = steinmetz_core_loss(req.k, req.f, req.b_peak, req.alpha, req.beta)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "core_loss_W_m3": loss,
        "inputs": req.model_dump(),
        "equation": "P = k · f^α · B^β",
    }


# ─── Bertotti loss separation ─────────────────────────────────────────────────

@router.post("/simulate/bertotti")
async def simulate_bertotti(req: BertottiRequest) -> dict:
    """
    Full Bertotti loss-separation model.
    """
    try:
        result = total_bertotti_loss(
            req.k_h, req.k_e, req.k_ex, req.f, req.b_peak, req.n
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"loss_components_W_m3": result, "inputs": req.model_dump()}


# ─── Steinmetz Exponent Regression Fitting ────────────────────────────────────

@router.post("/fit/steinmetz")
async def fit_steinmetz_coefficients(req: SteinmetzFitRequest) -> dict:
    """
    Fit Steinmetz coefficients dynamically from experimental data using Scikit-Learn.
    """
    import numpy as np
    f_arr = np.array([dp.frequency for dp in req.data_points])
    b_arr = np.array([dp.b_peak for dp in req.data_points])
    p_arr = np.array([dp.p_loss for dp in req.data_points])
    
    model = SteinmetzModel()
    try:
        model.fit(f_arr, b_arr, p_arr)
        coeffs = model.get_coefficients()
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Steinmetz fitting failed: {str(exc)}",
        ) from exc
        
    return {
        "k": coeffs["k"],
        "alpha": coeffs["alpha_frequency_exponent"],
        "beta": coeffs["beta_flux_density_exponent"],
        "message": "Ridge regression completed successfully."
    }


# ─── BH curve analysis ────────────────────────────────────────────────────────

@router.post("/analyse/bh-curve")
async def analyse_bh_curve(req: BHCurveRequest) -> dict:
    """
    Accept arrays of H and B values, return material parameters
    including knee point, permeability estimates, and hysteresis loop data.
    """
    import numpy as np

    h = np.array(req.h_values, dtype=np.float64)
    b = np.array(req.b_values, dtype=np.float64)

    if len(h) != len(b):
        raise HTTPException(
            status_code=422,
            detail="h_values and b_values must have the same length.",
        )
    if not (h > 0).all():
        raise HTTPException(status_code=422, detail="All H values must be positive for initial DC curve analysis.")

    analyser = BHCurveAnalyser()
    try:
        result = analyser.analyse(h, b)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"material": req.material_name, "analysis": result}


# ─── Hysteresis loop generation ───────────────────────────────────────────────

@router.post("/analyse/hysteresis-loop")
async def get_hysteresis_loop(req: HysteresisLoopRequest) -> dict:
    """
    Generate a parameterised symmetric hysteresis B-H loop.
    """
    loop = generate_hysteresis_loop(
        h_max=req.h_max,
        b_sat=req.b_sat,
        coercivity_h=req.coercivity_h,
        remanence_b=req.remanence_b,
        n_points=req.n_points,
        frequency=req.frequency,
    )
    
    # Calculate Loop Area to get dynamic Hysteresis loss
    area = calculate_loop_area(loop["h"], loop["b_upper"], loop["b_lower"])
    
    return {
        "hysteresis_loop": loop,
        "enclosed_area_J_m3": area
    }


# ─── CSV upload ───────────────────────────────────────────────────────────────

@router.post("/upload/bh-csv")
async def upload_bh_csv(
    file: UploadFile = File(...),
    material_name: str = "Uploaded Material",
    frequency_hz: float = 50.0,
    temperature_c: float = 25.0,
) -> dict:
    """
    Upload a CSV file with columns [H_A_per_m, B_Tesla].
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=415,
            detail="Only .csv files are accepted.",
        )

    csv_bytes = await file.read()
    if len(csv_bytes) > 5 * 1024 * 1024:   # 5 MB limit
        raise HTTPException(status_code=413, detail="File must be < 5 MB.")

    try:
        dataset = parse_csv_to_bh_dataset(
            csv_bytes, material_name, frequency_hz, temperature_c
        )
    except (ValueError, Exception) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    h_arr, b_arr = bh_dataset_to_arrays(dataset)
    analyser = BHCurveAnalyser()
    analysis = analyser.analyse(h_arr, b_arr)

    return {
        "filename": file.filename,
        "rows_parsed": len(dataset.data_points),
        "material": dataset.material_name,
        "frequency_Hz": dataset.frequency_Hz,
        "temperature_C": dataset.temperature_C,
        "analysis": analysis,
    }
