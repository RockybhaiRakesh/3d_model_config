import os
import sys
import json
import math
import zipfile
import tempfile
import trimesh
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request, Response, BackgroundTasks, status
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Bambu A1 Cloud Optimizer SaaS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_JSON_PATH = os.path.join(BASE_DIR, "bambu_base_template.json")

# Environment-driven Authentication (Defaults provided for quick start)
AUTH_USER = os.getenv("APP_AUTH_USER", "blogo")
AUTH_PASS = os.getenv("APP_AUTH_PASS", "Blogo@12")
AUTH_TOKEN = os.getenv("APP_AUTH_TOKEN", "blogo_secure_session_token_2026_a1")

def get_current_user(request: Request):
    token = request.cookies.get("blogo_token") or request.headers.get("X-Blogo-Auth")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            
    if token != AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Please log in with valid credentials."
        )
    return AUTH_USER

def get_base_bambu_template():
    if os.path.exists(TEMPLATE_JSON_PATH):
        try:
            with open(TEMPLATE_JSON_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def load_mesh_safe(file_path):
    loaded = trimesh.load(file_path)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No valid 3D geometry found in 3MF/Scene.")
        return trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        return loaded
    else:
        raise ValueError("Unsupported 3D geometry format.")

def generate_optimized_3mf(input_file_path, output_3mf_path, mode="crystal", scale_factor=1.0):
    mesh = load_mesh_safe(input_file_path)
    if abs(scale_factor - 1.0) > 1e-4:
        mesh.apply_scale(scale_factor)
        
    extents = mesh.extents
    width, depth, height = float(extents[0]), float(extents[1]), float(extents[2])
    
    cfg = get_base_bambu_template()
    for k, v in list(cfg.items()):
        if isinstance(v, list) and len(v) == 4:
            cfg[k] = [v[0]]
            
    cfg["flush_volumes_matrix"] = ["0"]
    cfg["flush_volumes_vector"] = ["0", "0"]
    
    cfg["printer_model"] = "Bambu Lab A1"
    cfg["printer_settings_id"] = "Bambu Lab A1 0.4 nozzle"
    cfg["print_compatible_printers"] = ["Bambu Lab A1 0.4 nozzle"]
    cfg["default_filament_profile"] = ["Bambu PLA Basic @BBL A1"]
    cfg["filament_settings_id"] = ["Bambu PLA Basic @BBL A1"]
    cfg["filament_type"] = ["PLA"]
    cfg["filament_vendor"] = ["Bambu Lab"]
    cfg["filament_colour"] = ["#000000"]
    cfg["default_filament_colour"] = ["#000000"]
    cfg["filament_ids"] = ["GFA00"]
    cfg["filament_self_index"] = ["1"]
    cfg["filament_map"] = ["1"]
    cfg["filament_volume_map"] = ["0"]
    cfg["filament_is_support"] = ["0"]
    cfg["curr_bed_type"] = "Textured PEI Plate"
    cfg["textured_plate_temp"] = ["65"]
    cfg["textured_plate_temp_initial_layer"] = ["65"]
    cfg["nozzle_temperature"] = ["220"]
    cfg["nozzle_temperature_initial_layer"] = ["220"]
    cfg["initial_layer_print_height"] = "0.2"
    cfg["seam_position"] = "back"
    cfg["apply_scarf_seam_on_circles"] = "1"
    cfg["scarf_seam_type"] = "contour_and_hole"
    cfg["scarf_length"] = "15"
    
    cfg["enable_support"] = "1"
    cfg["support_type"] = "tree(auto)"
    cfg["support_style"] = "tree_slim"
    cfg["independent_support_layer_height"] = "1"
    cfg["support_threshold_angle"] = "30"
    
    if height <= 50.0:
        cfg["tree_support_branch_diameter"] = "2.0"
        cfg["tree_support_tip_diameter"] = "0.5"
        cfg["support_top_z_distance"] = "0.16"
        cfg["support_bottom_z_distance"] = "0.16"
        cfg["support_object_xy_distance"] = "0.30"
    else:
        cfg["tree_support_branch_diameter"] = "3.0"
        cfg["tree_support_tip_diameter"] = "0.8"
        cfg["support_top_z_distance"] = "0.20"
        cfg["support_bottom_z_distance"] = "0.20"
        cfg["support_object_xy_distance"] = "0.45"

    if mode == "crystal":
        if height <= 50.0:
            layer_height = "0.08"
            preset_name = "0.08mm Extra Fine @BBL A1"
            outer_wall_speed = "35"
            inner_wall_speed = "80"
            sparse_infill_speed = "120"
            infill_density = "15%"
            wall_loops = 4
        elif height <= 130.0:
            layer_height = "0.12"
            preset_name = "0.12mm High Quality @BBL A1"
            outer_wall_speed = "45"
            inner_wall_speed = "120"
            sparse_infill_speed = "180"
            infill_density = "15%"
            wall_loops = 3
        else:
            layer_height = "0.16"
            preset_name = "0.16mm Optimal @BBL A1"
            outer_wall_speed = "65"
            inner_wall_speed = "150"
            sparse_infill_speed = "220"
            infill_density = "12%"
            wall_loops = 3
            
        cfg["wall_generator"] = "arachne"
        cfg["sparse_infill_pattern"] = "gyroid"
        
    elif mode == "mechanical":
        layer_height = "0.20"
        preset_name = "0.20mm Standard @BBL A1"
        outer_wall_speed = "120"
        inner_wall_speed = "180"
        sparse_infill_speed = "220"
        infill_density = "25%"
        wall_loops = 4
        cfg["wall_generator"] = "classic"
        cfg["sparse_infill_pattern"] = "gyroid"
        cfg["top_shell_layers"] = "6"
        cfg["bottom_shell_layers"] = "5"
        
    else:  # Standard / Fast
        layer_height = "0.20"
        preset_name = "0.20mm Standard @BBL A1"
        outer_wall_speed = "150"
        inner_wall_speed = "220"
        sparse_infill_speed = "270"
        infill_density = "15%"
        wall_loops = 2
        cfg["wall_generator"] = "arachne"
        cfg["sparse_infill_pattern"] = "cross_hatch"

    cfg["print_settings_id"] = preset_name
    cfg["default_print_profile"] = preset_name
    cfg["layer_height"] = layer_height
    cfg["outer_wall_speed"] = outer_wall_speed
    cfg["inner_wall_speed"] = inner_wall_speed
    cfg["sparse_infill_speed"] = sparse_infill_speed
    cfg["sparse_infill_density"] = infill_density
    cfg["wall_loops"] = wall_loops
    
    diff_keys = "enable_support;support_type;support_style;support_threshold_angle;layer_height;outer_wall_speed;inner_wall_speed;wall_generator;sparse_infill_pattern;wall_loops;seam_position;support_top_z_distance;support_object_xy_distance;"
    cfg["different_settings_to_system"] = [diff_keys, "", ""]
    
    bounds = mesh.bounds
    min_z = bounds[0][2]
    mesh.apply_translation([-mesh.centroid[0] + 128.0, -mesh.centroid[1] + 128.0, -min_z])
    
    temp_base_3mf = output_3mf_path + ".temp.3mf"
    mesh.export(temp_base_3mf)
    
    slice_info_xml = '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n  <header>\n    <header_item key="X-BBL-Client-Type" value="slicer"/>\n    <header_item key="X-BBL-Client-Version" value="02.09.01.70"/>\n  </header>\n</config>\n'
    model_settings_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<config>
  <object id="1">
    <metadata key="name" value="{os.path.basename(input_file_path)}"/>
    <metadata key="extruder" value="1"/>
    <part id="1" subtype="normal_part">
      <metadata key="name" value="{os.path.basename(input_file_path)}"/>
      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>
      <metadata key="extruder" value="1"/>
    </part>
  </object>
  <plate>
    <metadata key="plater_id" value="1"/>
    <metadata key="plater_name" value=""/>
    <metadata key="locked" value="false"/>
    <metadata key="filament_maps" value="1"/>
    <metadata key="filament_volume_maps" value="0"/>
    <model_instance>
      <metadata key="object_id" value="1"/>
      <metadata key="instance_id" value="0"/>
      <metadata key="identify_id" value="1"/>
    </model_instance>
  </plate>
</config>'''

    plate_1_data = {
        "bed_type": "textured_plate",
        "filament_colors": ["#000000"],
        "filament_ids": [0],
        "first_extruder": 0,
        "nozzle_diameter": 0.4
    }

    with zipfile.ZipFile(temp_base_3mf, 'r') as zin:
        with zipfile.ZipFile(output_3mf_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == '3D/3dmodel.model':
                    text = data.decode('utf-8', errors='ignore')
                    if 'xmlns:BambuStudio' not in text:
                        text = text.replace('<model ', '<model xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" ', 1)
                    if 'BambuStudio:3mfVersion' not in text:
                        m_start = text.find('<model')
                        m_end = text.find('>', m_start)
                        meta_tags = (
                            '\n <metadata name="Application">BambuStudio-02.04.00.70</metadata>'
                            '\n <metadata name="BambuStudio:3mfVersion">1</metadata>'
                        )
                        text = text[:m_end+1] + meta_tags + text[m_end+1:]
                    data = text.encode('utf-8')
                zout.writestr(item, data)
            zout.writestr('Metadata/project_settings.config', json.dumps(cfg, indent=4))
            zout.writestr('Metadata/model_settings.config', model_settings_xml)
            zout.writestr('Metadata/slice_info.config', slice_info_xml)
            zout.writestr('Metadata/plate_1.json', json.dumps(plate_1_data, indent=4))
            zout.writestr('Metadata/filament_sequence.json', '{"plate_1":{"sequence":[]}}')
            zout.writestr('Metadata/cut_information.xml', '<?xml version="1.0" encoding="utf-8"?>\n<objects>\n <object id="1">\n  <cut_id id="0" check_sum="1" connectors_cnt="0"/>\n </object>\n</objects>\n')

    if os.path.exists(temp_base_3mf):
        os.remove(temp_base_3mf)
        
    return {
        "width": round(width, 1),
        "depth": round(depth, 1),
        "height": round(height, 1),
        "layer_height": layer_height,
        "preset_name": preset_name
    }

# ================= AUTH ENDPOINTS =================
@app.post("/api/login")
async def login(request: Request, response: Response):
    try:
        body = await request.json()
        u = body.get("username", "").strip()
        p = body.get("password", "").strip()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request payload.")

    if u == AUTH_USER and p == AUTH_PASS:
        # Set auth cookie valid for 30 days
        response.set_cookie(
            key="blogo_token",
            value=AUTH_TOKEN,
            max_age=30 * 86400,
            httponly=False,  # Accessible by frontend JS for auth verification
            samesite="lax",
            secure=False     # Works on both http://localhost and https
        )
        return {"success": True, "token": AUTH_TOKEN, "username": AUTH_USER}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(key="blogo_token")
    return {"success": True, "message": "Logged out successfully."}

@app.get("/api/auth-check")
def auth_check(user: str = Depends(get_current_user)):
    return {"authenticated": True, "username": user}

@app.get("/api/health")
def health():
    return {"status": "ok", "app": "Bambu A1 Cloud Optimizer", "auth_enabled": True}

# ================= PROTECTED CORE ENDPOINTS =================
@app.post("/api/analyze")
async def analyze_model(
    file: UploadFile = File(...),
    user: str = Depends(get_current_user)
):
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in ['.stl', '.3mf', '.obj']:
        raise HTTPException(status_code=400, detail="Only STL, 3MF, and OBJ files are supported.")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        mesh = load_mesh_safe(tmp_path)
        extents = mesh.extents
        w, d, h = float(extents[0]), float(extents[1]), float(extents[2])
        faces = len(mesh.faces)
        vol = (w * d * h) / 1000.0 * 0.35
        if hasattr(mesh, 'is_volume') and mesh.is_volume:
            vol = mesh.volume / 1000.0
            
        est_weight = vol * 1.24 * 0.45
        est_cost = est_weight * 1.20
        
        return {
            "success": True,
            "filename": file.filename,
            "width": round(w, 1),
            "depth": round(d, 1),
            "height": round(h, 1),
            "faces": faces,
            "est_weight_g": round(max(1.0, est_weight), 1),
            "est_cost_rs": round(max(1.0, est_cost), 1),
            "tmp_path": tmp_path
        }
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_temp_files(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

@app.post("/api/optimize")
async def optimize_model(
    background_tasks: BackgroundTasks,
    tmp_path: str = Form(...),
    original_name: str = Form(...),
    mode: str = Form("crystal"),
    scale: float = Form(1.0),
    user: str = Depends(get_current_user)
):
    if not os.path.exists(tmp_path):
        raise HTTPException(status_code=400, detail="Temporary file expired. Please upload again.")
        
    base_name = os.path.splitext(original_name)[0]
    scale_str = f"_{int(scale*100)}pct" if abs(scale - 1.0) > 1e-4 else ""
    out_filename = f"{base_name}{scale_str}_BambuA1_Optimized.3mf"
    out_path = os.path.join(tempfile.gettempdir(), out_filename)
    
    try:
        generate_optimized_3mf(tmp_path, out_path, mode=mode, scale_factor=scale)
        background_tasks.add_task(cleanup_temp_files, tmp_path, out_path)
        return FileResponse(
            out_path,
            filename=out_filename,
            media_type="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

# ================= MODERN SAAS UI WITH LOGIN =================
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Bambu A1 Smart Cloud Optimizer | Blogo 3D Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Plus Jakarta Sans', sans-serif; }
    .glass-card { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(12px); }
  </style>
</head>
<body class="bg-gradient-to-br from-slate-900 via-teal-950 to-slate-900 min-h-screen text-slate-800 flex flex-col justify-between">

  <!-- Header -->
  <header class="border-b border-teal-500/20 bg-slate-950/70 backdrop-blur sticky top-0 z-50">
    <div class="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-teal-500 to-emerald-400 flex items-center justify-center text-white font-extrabold shadow-lg shadow-teal-500/30 text-xl">
          ⚡
        </div>
        <div>
          <h1 class="text-white font-bold text-lg leading-tight">Bambu A1 Smart Optimizer</h1>
          <p class="text-teal-400 text-xs font-medium">Blogo Studio &bull; Private Production Engine</p>
        </div>
      </div>
      <div class="flex items-center space-x-3">
        <div id="auth-status-bar" class="hidden flex items-center space-x-3">
          <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            🟢 Online (A1 Calibrated)
          </span>
          <span class="text-xs text-slate-300 bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-700">
            👤 <span id="logged-username" class="font-bold text-teal-300">blogo</span>
          </span>
          <button onclick="handleLogout()" class="text-xs text-rose-300 hover:text-white bg-rose-500/20 hover:bg-rose-600 px-2.5 py-1 rounded-md border border-rose-500/30 transition">
            Logout
          </button>
        </div>
        <div id="auth-login-pill" class="flex items-center space-x-2">
          <span class="text-xs text-amber-300 bg-amber-500/10 px-3 py-1 rounded-full border border-amber-500/30">
            🔒 Private Access
          </span>
        </div>
      </div>
    </div>
  </header>

  <!-- LOGIN SECTION (Shown when not authenticated) -->
  <div id="login-section" class="flex-grow flex items-center justify-center px-4 py-12">
    <div class="w-full max-w-md bg-slate-950/80 border border-teal-500/30 backdrop-blur-xl rounded-3xl p-8 shadow-2xl shadow-teal-950/80">
      <div class="text-center mb-6">
        <div class="w-16 h-16 rounded-2xl bg-gradient-to-tr from-teal-500 to-emerald-400 mx-auto flex items-center justify-center text-white text-3xl font-black shadow-lg shadow-teal-500/40 mb-3">
          ⚡
        </div>
        <h2 class="text-2xl font-black text-white">Blogo 3D Studio</h2>
        <p class="text-sm text-slate-400 mt-1">Sign in to access Bambu A1 Cloud Optimizer</p>
      </div>

      <div id="login-error" class="hidden mb-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs font-semibold text-center">
      </div>

      <form onsubmit="handleLogin(event)" class="space-y-4">
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1.5 uppercase tracking-wider">Username</label>
          <div class="relative">
            <span class="absolute inset-y-0 left-0 pl-3.5 flex items-center text-slate-400 text-sm">👤</span>
            <input type="text" id="login-username" value="blogo" required
              class="w-full pl-10 pr-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-white text-sm focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition"
              placeholder="Enter username" />
          </div>
        </div>

        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1.5 uppercase tracking-wider">Password</label>
          <div class="relative">
            <span class="absolute inset-y-0 left-0 pl-3.5 flex items-center text-slate-400 text-sm">🔑</span>
            <input type="password" id="login-password" required
              class="w-full pl-10 pr-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-white text-sm focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition"
              placeholder="Enter password" />
          </div>
        </div>

        <button type="submit" id="login-btn"
          class="w-full mt-2 py-3 px-4 rounded-xl bg-gradient-to-r from-teal-500 to-emerald-500 hover:from-teal-400 hover:to-emerald-400 text-white font-bold text-sm shadow-lg shadow-teal-500/30 transition transform hover:-translate-y-0.5 active:translate-y-0">
          Sign In
        </button>
      </form>

      <div class="mt-6 text-center text-slate-500 text-xs">
        Bambu Lab A1 Official Compatible 3MF Engine &bull; Private v2.5
      </div>
    </div>
  </div>

  <!-- MAIN APP CONTAINER (Shown when authenticated) -->
  <main id="app-section" class="hidden max-w-5xl mx-auto px-4 py-8 flex-grow w-full">
    
    <!-- Hero Banner -->
    <div class="text-center mb-8">
      <div class="inline-block px-4 py-1.5 rounded-full bg-teal-500/10 border border-teal-500/20 text-teal-300 text-xs font-semibold mb-3">
        ✨ 1-Click Zero Failure 3MF Generation for Bambu Lab A1
      </div>
      <h2 class="text-3xl md:text-4xl font-extrabold text-white tracking-tight">
        Drop Your Model. Download Ready-To-Print 3MF.
      </h2>
      <p class="text-slate-400 text-sm mt-2 max-w-2xl mx-auto">
        Automatically applies <b>Arachne wall generation</b>, <b>Tree Slim Auto supports</b>, <b>0.16mm Z-contact gap</b>, and <b>65°C PEI bed adhesion</b> directly into official Bambu Studio package structure.
      </p>
    </div>

    <!-- Drag & Drop Upload Zone -->
    <div id="drop-zone" class="border-2 border-dashed border-teal-500/40 hover:border-teal-400 bg-slate-900/40 hover:bg-slate-900/70 rounded-3xl p-10 text-center cursor-pointer transition-all duration-200 shadow-xl backdrop-blur-md mb-8">
      <input type="file" id="file-input" class="hidden" accept=".stl,.3mf,.obj" onchange="handleFileSelect(event)">
      
      <div id="upload-prompt" class="space-y-4">
        <div class="w-20 h-20 bg-teal-500/10 border border-teal-500/20 rounded-2xl flex items-center justify-center mx-auto text-4xl text-teal-400">
          📥
        </div>
        <div>
          <p class="text-lg font-bold text-white">Drag & drop your STL / 3MF / OBJ file here</p>
          <p class="text-xs text-slate-400 mt-1">or click to browse from your computer (Up to 150MB)</p>
        </div>
        <div class="flex items-center justify-center space-x-2 text-xs text-teal-400/80 font-medium pt-2">
          <span>✓ Devotional Idols</span>
          <span>&bull;</span>
          <span>✓ Miniatures</span>
          <span>&bull;</span>
          <span>✓ Mechanical Parts</span>
        </div>
      </div>

      <!-- Loading Spinner -->
      <div id="upload-loading" class="hidden py-8 space-y-4">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-teal-400 mx-auto"></div>
        <p class="text-sm font-semibold text-teal-300" id="loading-status">Analyzing geometry & computing dimensions...</p>
      </div>
    </div>

    <!-- Optimization Panel (Hidden until uploaded) -->
    <div id="optimizer-panel" class="hidden glass-card rounded-3xl p-6 md:p-8 shadow-2xl border border-white/40 space-y-6">
      
      <!-- Model Summary -->
      <div class="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-slate-200/80 gap-4">
        <div>
          <div class="flex items-center space-x-2">
            <span class="px-2.5 py-1 bg-teal-100 text-teal-800 text-xs font-bold rounded-md uppercase" id="badge-filetype">STL</span>
            <h3 class="text-xl font-extrabold text-slate-800" id="filename-display">lord_ayyappan.stl</h3>
          </div>
          <p class="text-xs text-slate-500 mt-1" id="dims-display">Active Dimensions: Width: 0mm | Depth: 0mm | Height: 0mm</p>
          <p class="text-xs font-semibold text-emerald-600 mt-0.5" id="cost-display">Est Material: ~0g | Est Cost: ₹0</p>
        </div>

        <button onclick="resetUpload()" class="text-xs font-bold text-slate-500 hover:text-slate-800 underline self-start md:self-center">
          ✕ Change File
        </button>
      </div>

      <!-- Slicing Mode Selection -->
      <div>
        <label class="block text-sm font-bold text-slate-800 mb-2">1. Select Print Precision Mode</label>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
          
          <label class="relative flex flex-col p-4 border-2 border-teal-500 bg-teal-50/40 rounded-2xl cursor-pointer hover:bg-teal-50/70 transition">
            <input type="radio" name="mode" value="crystal" checked class="absolute top-4 right-4 text-teal-600 focus:ring-teal-500">
            <span class="text-sm font-black text-slate-800">👑 Crystal Devotional</span>
            <span class="text-xs text-teal-700 font-semibold mt-0.5">Ultra Sharp Face & Hand Details</span>
            <span class="text-[11px] text-slate-500 mt-2">Adaptive 0.08mm - 0.12mm layers, 35mm/s outer wall, Arachne engine.</span>
          </label>

          <label class="relative flex flex-col p-4 border-2 border-slate-200 bg-white rounded-2xl cursor-pointer hover:bg-slate-50 transition">
            <input type="radio" name="mode" value="mechanical" class="absolute top-4 right-4 text-teal-600 focus:ring-teal-500">
            <span class="text-sm font-black text-slate-800">⚙️ Mechanical / Strong</span>
            <span class="text-xs text-indigo-700 font-semibold mt-0.5">High Strength & Fast</span>
            <span class="text-[11px] text-slate-500 mt-2">0.20mm layers, 4 walls, 25% gyroid infill, 6 top/bottom shells.</span>
          </label>

          <label class="relative flex flex-col p-4 border-2 border-slate-200 bg-white rounded-2xl cursor-pointer hover:bg-slate-50 transition">
            <input type="radio" name="mode" value="standard" class="absolute top-4 right-4 text-teal-600 focus:ring-teal-500">
            <span class="text-sm font-black text-slate-800">⚡ Standard Speed</span>
            <span class="text-xs text-amber-700 font-semibold mt-0.5">Everyday Prototyping</span>
            <span class="text-[11px] text-slate-500 mt-2">0.20mm layers, standard walls, 15% infill, optimized high-speed.</span>
          </label>

        </div>
      </div>

      <!-- Quick Scaling Tools -->
      <div>
        <div class="flex items-center justify-between mb-2">
          <label class="text-sm font-bold text-slate-800">2. Adjust Idol Height & Scale</label>
          <span class="text-xs font-bold text-teal-700 bg-teal-100 px-2.5 py-0.5 rounded-full" id="active-scale-badge">100% Scale</span>
        </div>
        
        <div class="flex flex-wrap items-center gap-2 mb-3">
          <button onclick="setScale(1.0)" class="px-3 py-1.5 text-xs font-bold bg-slate-100 hover:bg-slate-200 rounded-lg text-slate-700">Original (100%)</button>
          <button onclick="setScale(0.8)" class="px-3 py-1.5 text-xs font-bold bg-slate-100 hover:bg-slate-200 rounded-lg text-slate-700">80%</button>
          <button onclick="setScale(0.6)" class="px-3 py-1.5 text-xs font-bold bg-slate-100 hover:bg-slate-200 rounded-lg text-slate-700">60%</button>
          <button onclick="setScale(0.5)" class="px-3 py-1.5 text-xs font-bold bg-slate-100 hover:bg-slate-200 rounded-lg text-slate-700">50% Pocket Idol</button>
          <button onclick="fitBed()" class="px-3 py-1.5 text-xs font-bold bg-amber-100 hover:bg-amber-200 rounded-lg text-amber-800">Fit Max Bed (220mm)</button>
        </div>

        <div class="flex items-center space-x-3 max-w-sm">
          <div class="relative flex-grow">
            <input type="number" id="height-input" step="1.0" class="w-full px-3 py-2 border border-slate-300 rounded-xl text-sm font-bold text-slate-800 focus:outline-none focus:border-teal-500" placeholder="Target Height">
            <span class="absolute right-3 top-2 text-xs font-semibold text-slate-400">mm Height</span>
          </div>
          <button onclick="applyHeight()" class="px-4 py-2 bg-slate-800 hover:bg-slate-900 text-white text-xs font-bold rounded-xl transition">
            Set
          </button>
        </div>
      </div>

      <!-- Action Button -->
      <div class="pt-4 border-t border-slate-200">
        <button id="download-btn" onclick="downloadOptimized3MF()" class="w-full py-4 px-6 rounded-2xl bg-gradient-to-r from-teal-500 via-emerald-500 to-teal-600 hover:from-teal-400 hover:to-emerald-400 text-white font-extrabold text-base shadow-xl shadow-teal-500/30 flex items-center justify-center space-x-2 transition transform hover:-translate-y-0.5 active:translate-y-0">
          <span>🚀 Generate & Download 1-Click Bambu 3MF</span>
        </button>
        <p class="text-center text-xs text-slate-500 mt-2 font-medium">
          Directly opens in Bambu Studio / OrcaSlicer without geometry or configuration warnings.
        </p>
      </div>

    </div>

  </main>

  <!-- Footer -->
  <footer class="border-t border-teal-500/20 bg-slate-950/80 py-4 text-center text-xs text-slate-400">
    <div class="max-w-6xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
      <p>&copy; 2026 Blogo 3D Studio &bull; Bambu Lab A1 Precision Slicing Engine</p>
      <div class="flex items-center space-x-4">
        <span>Arachne Wall Gen</span>
        <span>&bull;</span>
        <span>Tree Slim Support</span>
        <span>&bull;</span>
        <span>Scarf Seam Active</span>
      </div>
    </div>
  </footer>

  <script>
    let currentTmpPath = "";
    let originalFilename = "";
    let baseWidth = 0, baseDepth = 0, baseHeight = 0;
    let currentScale = 1.0;

    // Check Auth on page load
    async function checkAuth() {
      try {
        const res = await fetch('/api/auth-check');
        if (res.ok) {
          const data = await res.json();
          showApp(data.username || 'blogo');
        } else {
          showLogin();
        }
      } catch (e) {
        showLogin();
      }
    }

    function showApp(username) {
      document.getElementById('login-section').classList.add('hidden');
      document.getElementById('app-section').classList.remove('hidden');
      document.getElementById('auth-status-bar').classList.remove('hidden');
      document.getElementById('auth-login-pill').classList.add('hidden');
      document.getElementById('logged-username').innerText = username;
    }

    function showLogin() {
      document.getElementById('login-section').classList.remove('hidden');
      document.getElementById('app-section').classList.add('hidden');
      document.getElementById('auth-status-bar').classList.add('hidden');
      document.getElementById('auth-login-pill').classList.remove('hidden');
    }

    async function handleLogin(e) {
      e.preventDefault();
      const u = document.getElementById('login-username').value.trim();
      const p = document.getElementById('login-password').value.trim();
      const errBox = document.getElementById('login-error');
      const btn = document.getElementById('login-btn');

      errBox.classList.add('hidden');
      btn.disabled = true;
      btn.innerText = "Signing in...";

      try {
        const res = await fetch('/api/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: u, password: p })
        });

        if (res.ok) {
          const data = await res.json();
          showApp(data.username);
        } else {
          const err = await res.json();
          errBox.innerText = err.detail || 'Invalid username or password';
          errBox.classList.remove('hidden');
        }
      } catch (err) {
        errBox.innerText = 'Connection error. Please try again.';
        errBox.classList.remove('hidden');
      } finally {
        btn.disabled = false;
        btn.innerText = "Sign In";
      }
    }

    async function handleLogout() {
      try {
        await fetch('/api/logout', { method: 'POST' });
      } catch (e) {}
      showLogin();
    }

    // Initialize check
    checkAuth();

    // Drag and Drop Setup
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('border-teal-400', 'bg-slate-900/80'); });
    dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('border-teal-400', 'bg-slate-900/80'); });
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('border-teal-400', 'bg-slate-900/80');
      if (e.dataTransfer.files.length > 0) {
        uploadFile(e.dataTransfer.files[0]);
      }
    });

    function handleFileSelect(event) {
      if (event.target.files.length > 0) {
        uploadFile(event.target.files[0]);
      }
    }

    function resetUpload() {
      document.getElementById('optimizer-panel').classList.add('hidden');
      document.getElementById('drop-zone').classList.remove('hidden');
      document.getElementById('upload-loading').classList.add('hidden');
      document.getElementById('upload-prompt').classList.remove('hidden');
      fileInput.value = "";
    }

    async function uploadFile(file) {
      document.getElementById('upload-prompt').classList.add('hidden');
      document.getElementById('upload-loading').classList.remove('hidden');
      
      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/api/analyze', {
          method: 'POST',
          body: formData
        });

        if (res.status === 401) {
          showLogin();
          throw new Error("Session expired. Please log in again.");
        }

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Analysis failed');
        }

        const data = await res.json();
        currentTmpPath = data.tmp_path;
        originalFilename = data.filename;
        baseWidth = data.width;
        baseDepth = data.depth;
        baseHeight = data.height;
        currentScale = 1.0;

        document.getElementById('filename-display').innerText = originalFilename;
        document.getElementById('badge-filetype').innerText = originalFilename.split('.').pop().toUpperCase();
        document.getElementById('height-input').value = baseHeight.toFixed(1);

        updateDimsUI();

        document.getElementById('drop-zone').classList.add('hidden');
        document.getElementById('optimizer-panel').classList.remove('hidden');

      } catch (err) {
        alert('Error: ' + err.message);
        document.getElementById('upload-loading').classList.add('hidden');
        document.getElementById('upload-prompt').classList.remove('hidden');
      }
    }

    function setScale(factor) {
      currentScale = factor;
      document.getElementById('height-input').value = (baseHeight * currentScale).toFixed(1);
      updateDimsUI();
    }

    function applyHeight() {
      const targetH = parseFloat(document.getElementById('height-input').value);
      if (targetH > 0 && baseHeight > 0) {
        currentScale = targetH / baseHeight;
        updateDimsUI();
      }
    }

    function fitBed() {
      const maxBed = 220.0;
      const scaleX = maxBed / baseWidth;
      const scaleY = maxBed / baseDepth;
      const scaleZ = maxBed / baseHeight;
      currentScale = Math.min(scaleX, scaleY, scaleZ);
      document.getElementById('height-input').value = (baseHeight * currentScale).toFixed(1);
      updateDimsUI();
    }

    function updateDimsUI() {
      const w = (baseWidth * currentScale).toFixed(1);
      const d = (baseDepth * currentScale).toFixed(1);
      const h = (baseHeight * currentScale).toFixed(1);

      document.getElementById('active-scale-badge').innerText = `${(currentScale * 100).toFixed(0)}% Scale`;
      document.getElementById('dims-display').innerText = `Active Dimensions: Width: ${w} mm | Depth: ${d} mm | Height: ${h} mm`;

      const vol = (w * d * h) / 1000.0 * 0.35;
      const weight = (vol * 1.24 * 0.45).toFixed(1);
      const cost = (weight * 1.20).toFixed(0);
      document.getElementById('cost-display').innerText = `📊 Est Material: ~${weight}g | Est PLA Cost: ~₹${cost}`;
    }

    async function downloadOptimized3MF() {
      const btn = document.getElementById('download-btn');
      btn.disabled = true;
      btn.innerText = "⏳ Packaging Bambu A1 3MF...";

      const mode = document.querySelector('input[name="mode"]:checked').value;
      const formData = new FormData();
      formData.append('tmp_path', currentTmpPath);
      formData.append('original_name', originalFilename);
      formData.append('mode', mode);
      formData.append('scale', currentScale);

      try {
        const res = await fetch('/api/optimize', { method: 'POST', body: formData });
        if (res.status === 401) {
          showLogin();
          throw new Error("Session expired. Please log in again.");
        }
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Download failed');
        }

        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const scaleStr = Math.abs(currentScale - 1.0) > 1e-4 ? `_${Math.round(currentScale*100)}pct` : '';
        const base = originalFilename.substring(0, originalFilename.lastIndexOf('.')) || originalFilename;
        a.download = `${base}${scaleStr}_BambuA1_Optimized.3mf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      } catch (err) {
        alert('Error: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerText = "🚀 Generate & Download 1-Click Bambu 3MF";
      }
    }
  </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("web_app:app", host=host, port=port, reload=True)
