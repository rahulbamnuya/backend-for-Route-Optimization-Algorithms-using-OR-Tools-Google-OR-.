import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
import openrouteservice
from openrouteservice.exceptions import ApiError
from fastapi.encoders import jsonable_encoder

# Import solver functions
from .vrp_solver import (
    get_distance_matrix,
    solve_cvrp_without_restrictions,
    compute_route_geometries,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- Load environment variables ---
load_dotenv()
ORS_API_KEY = os.getenv("ORS_API_KEY")

# --- FastAPI App ---
app = FastAPI(title="VRP Solver API")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class Location(BaseModel):
    name: str
    latitude: float
    longitude: float

class Vehicle(BaseModel):
    id: str
    capacity: int
    fuel_cost_per_km: Optional[float] = None
    driver_cost_per_km: Optional[float] = None
    type: Optional[str] = None

class OptimizeRequest(BaseModel):
    locations: List[Location] = Field(
        ..., min_items=2, description="List of locations. First one MUST be the depot."
    )
    vehicles: List[Vehicle] = Field(..., min_items=1)
    demands: List[int]
    include_geometry: Optional[bool] = True
    time_limit_seconds: Optional[int] = 15

# --- Routes ---
@app.get("/")
def home():
    return {"message": "Route Optimization Backend is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/optimize")
def optimize(req: OptimizeRequest):
    # 1. Validate API Key
    if not ORS_API_KEY:
        logging.error("ORS_API_KEY not found in environment variables.")
        raise HTTPException(
            status_code=500, detail="Server configuration error: ORS_API_KEY not set."
        )

    # 2. Validate demands
    if len(req.demands) != len(req.locations):
        raise HTTPException(
            status_code=400,
            detail="The number of demands must match the number of locations.",
        )

    if req.demands[0] != 0:
        raise HTTPException(
            status_code=400,
            detail="The demand for the first location (depot) must be 0.",
        )

    # Log inputs
    try:
        vehicle_capacities = [v.capacity for v in req.vehicles]
        logging.info("--- VRP Optimization Request ---")
        logging.info(f"Locations: {len(req.locations)}")
        logging.info(f"Vehicles: {len(req.vehicles)}")
        logging.info(f"Demands: {req.demands}")
        logging.info(f"Vehicle Capacities: {vehicle_capacities}")
        logging.info(f"Total Demand: {sum(req.demands)}")
        logging.info(f"Total Capacity: {sum(vehicle_capacities)}")
        logging.info("--------------------------------")
    except Exception as e:
        logging.error(f"Error logging input: {e}")

    # 3. Infeasibility check
    max_vehicle_capacity = max(v.capacity for v in req.vehicles)
    for i, demand in enumerate(req.demands):
        if demand > max_vehicle_capacity:
            location_name = req.locations[i].name
            error_detail = (
                f"Infeasible: Demand at '{location_name}' ({demand}) "
                f"exceeds largest vehicle capacity ({max_vehicle_capacity})."
            )
            logging.warning(error_detail)
            raise HTTPException(status_code=400, detail=error_detail)

    # 4. Solve VRP
    try:
        client = openrouteservice.Client(key=ORS_API_KEY)

        # FIX: ORS expects (longitude, latitude)
        locations_coords = [(loc.longitude, loc.latitude) for loc in req.locations]
        location_names = [loc.name for loc in req.locations]
        vehicles_dict = jsonable_encoder(req.vehicles)

        logging.info("Fetching distance matrix from ORS...")
        distance_matrix = get_distance_matrix(client, locations_coords)
        if distance_matrix is None:
            raise HTTPException(
                status_code=503,
                detail="Failed to build distance matrix from ORS (service unavailable).",
            )

        logging.info("Solving VRP...")
        optimized = solve_cvrp_without_restrictions(
            distance_matrix,
            vehicles_dict,
            req.demands,
            location_names,
            time_limit_seconds=req.time_limit_seconds,
        )

        if not optimized:
            error_detail = (
                "No solution found. Either infeasible or solver timed out."
            )
            logging.warning(error_detail)
            raise HTTPException(status_code=500, detail=error_detail)

        logging.info(f"Solution found for {len(optimized)} vehicle(s).")

        if req.include_geometry:
            logging.info("Computing route geometries...")
            optimized = compute_route_geometries(client, locations_coords, optimized)

    except ApiError as e:
        logging.error(f"ORS API Error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Error communicating with OpenRouteService API: {e}",
        )
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Unexpected server error: {e}"
        )

    # 5. Return result
    logging.info("Optimization complete.")
    return jsonable_encoder(
        {
            "result": optimized,
            "summary": {
                "vehicles_used": len(optimized),
                "total_distance_km": round(
                    sum(v.get("Distance (km)", 0) for v in optimized), 2
                ),
                "total_load": sum(v.get("Load Carried", 0) for v in optimized),
            },
        }
    )
