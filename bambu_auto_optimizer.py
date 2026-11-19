import os
import sys
import json
import zipfile
import subprocess
import trimesh
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

BAMBU_STUDIO_PATH = r"C:\Program Files\Bambu Studio\bambu-studio.exe"
MAX_BED_SIZE = 256.0  # Bambu Lab A1 build limit in mm
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_JSON_PATH = os.path.join(SCRIPT_DIR, "bambu_base_template.json")
ICON_PATH = os.path.join(SCRIPT_DIR, "app_icon.ico")

def get_base_bambu_template():
    """
    Loads full 502-key official Bambu Studio project configuration template.
    """
    if os.path.exists(TEMPLATE_JSON_PATH):
        try:
            with open(TEMPLATE_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def load_mesh_safe(file_path):
    """
    Safely loads STL, 3MF, OBJ, etc. and returns a single concatenated Trimesh object.
    """
    loaded = trimesh.load(file_path)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    return mesh

def analyze_and_optimize_settings(mesh, mode="crystal"):
    """
    Dynamically tunes settings based on model bounding box & user intent.
    """
    extents = mesh.extents  # [dx, dy, dz]
    width, depth, height = float(extents[0]), float(extents[1]), float(extents[2])
    
    cfg = get_base_bambu_template()
    
    # Single filament cleanup (eliminate 4-color AMS clutter)
    for k, v in list(cfg.items()):
        if isinstance(v, list) and len(v) == 4:
            cfg[k] = [v[0]]
            
    cfg["flush_volumes_matrix"] = ["0"]
    cfg["flush_volumes_vector"] = ["0", "0"]
    
    # Base hardware & filament settings (Bambu Lab A1 + Bambu PLA Basic)
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
    
    # Support settings - MUST be string "1" for Bambu Studio to check the box
    cfg["enable_support"] = "1"
    cfg["support_type"] = "tree(auto)"
    cfg["support_style"] = "tree_slim"
    cfg["independent_support_layer_height"] = "1"
    cfg["support_threshold_angle"] = "30"
    
    # Micro-tuned tree supports based on scale
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
            category = "Miniature (< 50mm) - Ultra Fine Sharpness"
        elif height <= 130.0:
            layer_height = "0.12"
            preset_name = "0.12mm High Quality @BBL A1"
            outer_wall_speed = "45"
            inner_wall_speed = "120"
            sparse_infill_speed = "180"
            infill_density = "15%"
            wall_loops = 3
            category = "Devotional Idol / Statue (50-130mm) - Crystal Clear"
        else:
            layer_height = "0.16"
            preset_name = "0.16mm Optimal @BBL A1"
            outer_wall_speed = "65"
            inner_wall_speed = "150"
            sparse_infill_speed = "220"
            infill_density = "12%"
            wall_loops = 3
            category = "Large Statue (> 130mm) - Balanced Quality & Time"
            
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
        category = "Mechanical / Snap-Fit Part - High Strength"
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
        category = "Standard Quality - Everyday Fast Print"
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
    
    vol = (width * depth * height) / 1000.0 * 0.35
    if hasattr(mesh, 'is_volume') and mesh.is_volume:
        vol = mesh.volume / 1000.0
        
    est_weight_g = vol * 1.24 * 0.45  # PLA density 1.24, with shell/infill factor
    est_cost_rs = est_weight_g * 1.20  # ~Rs 1200 per 1kg Bambu PLA spool
    
    # Rough time estimation
    total_layers = int(height / float(layer_height))
    est_hours = (total_layers * 11) / 3600.0 + 0.35  # Approx 11 sec per layer on A1
    
    analysis = {
        "width": width,
        "depth": depth,
        "height": height,
        "volume_cm3": vol,
        "est_weight_g": max(1.0, est_weight_g),
        "est_cost_rs": max(1.0, est_cost_rs),
        "est_time_str": f"{int(est_hours)}h {int((est_hours % 1)*60)}m",
        "category": category,
        "preset_name": preset_name,
        "layer_height": layer_height,
        "wall_loops": wall_loops,
        "infill": f"{infill_density} {cfg['sparse_infill_pattern'].capitalize()}",
        "outer_wall_speed": f"{outer_wall_speed} mm/s",
        "wall_generator": cfg["wall_generator"].capitalize()
    }
    
    return cfg, analysis

def generate_bambu_3mf(input_file_path, output_3mf_path, mode="crystal", scale_factor=1.0):
    """
    Handles scaling, centering, and Bambu A1 project packaging for STL, 3MF, etc.
    """
    mesh = load_mesh_safe(input_file_path)
    
    if abs(scale_factor - 1.0) > 1e-4:
        mesh.apply_scale(scale_factor)
        
    cfg, analysis = analyze_and_optimize_settings(mesh, mode)
    
    is_3mf = input_file_path.lower().endswith(".3mf")
    
    if is_3mf and abs(scale_factor - 1.0) <= 1e-4:
        plate_1_data = {
            "bed_type": "textured_plate",
            "filament_colors": ["#000000"],
            "filament_ids": [0],
            "first_extruder": 0,
            "nozzle_diameter": 0.4
        }
        with zipfile.ZipFile(input_file_path, 'r') as zin:
            existing_cfg = {}
            if 'Metadata/project_settings.config' in zin.namelist():
                try:
                    existing_cfg = json.loads(zin.read('Metadata/project_settings.config').decode('utf-8'))
                except Exception:
                    pass
            existing_cfg.update(cfg)
            
            with zipfile.ZipFile(output_3mf_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    data = zin.read(item.filename)
                    if item.filename == 'Metadata/project_settings.config':
                        zout.writestr(item, json.dumps(existing_cfg, indent=4))
                    elif item.filename == 'Metadata/plate_1.json':
                        zout.writestr(item, json.dumps(plate_1_data, indent=4))
                    else:
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
                if 'Metadata/project_settings.config' not in zin.namelist():
                    zout.writestr('Metadata/project_settings.config', json.dumps(cfg, indent=4))
                if 'Metadata/plate_1.json' not in zin.namelist():
                    zout.writestr('Metadata/plate_1.json', json.dumps(plate_1_data, indent=4))
                if 'Metadata/filament_sequence.json' not in zin.namelist():
                    zout.writestr('Metadata/filament_sequence.json', '{"plate_1":{"sequence":[]}}')
                if 'Metadata/cut_information.xml' not in zin.namelist():
                    zout.writestr('Metadata/cut_information.xml', '<?xml version="1.0" encoding="utf-8"?>\n<objects>\n <object id="1">\n  <cut_id id="0" check_sum="1" connectors_cnt="0"/>\n </object>\n</objects>\n')
    else:
        # Re-center scaled mesh on Bambu A1 bed (256x256, center at 128, 128)
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
            
    return analysis

class BambuOptimizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bambu A1 Smart Print Optimizer")
        self.root.geometry("680x850")
        self.root.minsize(640, 800)
        self.root.configure(bg="#F4F6F9")
        
        if os.path.exists(ICON_PATH):
            try:
                self.root.iconbitmap(ICON_PATH)
            except Exception:
                pass
        
        self.input_file = tk.StringVar()
        self.mode_var = tk.StringVar(value="crystal")
        self.status_var = tk.StringVar(value="Ready. Select any STL or 3MF file to begin.")
        
        self.orig_mesh = None
        self.current_scale = 1.0
        
        self.target_height_var = tk.StringVar(value="")
        self.scale_pct_var = tk.StringVar(value="100")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Header Frame
        header = tk.Frame(self.root, bg="#008053", padx=20, pady=12)
        header.pack(fill="x")
        
        title = tk.Label(header, text="🎯 Bambu A1 Smart Print Optimizer", font=("Segoe UI", 18, "bold"), fg="white", bg="#008053")
        title.pack(anchor="w")
        subtitle = tk.Label(header, text="Auto 1-Click Crystal Clear Slicing Presets for Bambu Lab A1 + PLA Basic", font=("Segoe UI", 10), fg="#E0F2E9", bg="#008053")
        subtitle.pack(anchor="w")
        
        main_frame = tk.Frame(self.root, bg="#F4F6F9", padx=18, pady=10)
        main_frame.pack(fill="both", expand=True)
        
        # 1. File Selection Card
        file_card = tk.LabelFrame(main_frame, text=" 1. Select 3D Model (MakerWorld / STL / 3MF) ", font=("Segoe UI", 10, "bold"), bg="white", fg="#2C3E50", padx=12, pady=10)
        file_card.pack(fill="x", pady=(0, 10))
        
        entry = tk.Entry(file_card, textvariable=self.input_file, font=("Segoe UI", 10), bg="#F8F9FA", relief="solid", bd=1)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 10), ipady=4)
        
        btn_browse = tk.Button(file_card, text="📂 Browse Model", command=self.browse_file, bg="#008053", fg="white", font=("Segoe UI", 9, "bold"), relief="flat", padx=14, pady=4, cursor="hand2")
        btn_browse.pack(side="right")
        
        # 2. Size & Scaling Controls Card
        size_card = tk.LabelFrame(main_frame, text=" 2. Model Size & Scaling (Scale Anytime) ", font=("Segoe UI", 10, "bold"), bg="white", fg="#2C3E50", padx=12, pady=10)
        size_card.pack(fill="x", pady=(0, 10))
        
        self.lbl_orig_dims = tk.Label(size_card, text="Original Size: No model loaded", font=("Segoe UI", 9), bg="white", fg="#666")
        self.lbl_orig_dims.pack(anchor="w", pady=(0, 4))
        
        scale_input_frame = tk.Frame(size_card, bg="white")
        scale_input_frame.pack(fill="x", pady=4)
        
        tk.Label(scale_input_frame, text="Target Height (Z):", font=("Segoe UI", 9, "bold"), bg="white").pack(side="left", padx=(0, 4))
        ent_h = tk.Entry(scale_input_frame, textvariable=self.target_height_var, width=8, font=("Segoe UI", 9), bg="#F8F9FA", relief="solid", bd=1)
        ent_h.pack(side="left", padx=(0, 4), ipady=2)
        tk.Label(scale_input_frame, text="mm", font=("Segoe UI", 9), bg="white").pack(side="left", padx=(0, 10))
        
        btn_apply_h = tk.Button(scale_input_frame, text="Set Height", command=self.apply_target_height, bg="#34495E", fg="white", font=("Segoe UI", 8, "bold"), relief="flat", padx=8, pady=2, cursor="hand2")
        btn_apply_h.pack(side="left", padx=(0, 18))
        
        tk.Label(scale_input_frame, text="Scale %:", font=("Segoe UI", 9, "bold"), bg="white").pack(side="left", padx=(0, 4))
        ent_pct = tk.Entry(scale_input_frame, textvariable=self.scale_pct_var, width=6, font=("Segoe UI", 9), bg="#F8F9FA", relief="solid", bd=1)
        ent_pct.pack(side="left", padx=(0, 4), ipady=2)
        tk.Label(scale_input_frame, text="%", font=("Segoe UI", 9), bg="white").pack(side="left", padx=(0, 10))
        
        btn_apply_pct = tk.Button(scale_input_frame, text="Apply %", command=self.apply_scale_pct, bg="#34495E", fg="white", font=("Segoe UI", 8, "bold"), relief="flat", padx=8, pady=2, cursor="hand2")
        btn_apply_pct.pack(side="left")
        
        presets_frame = tk.Frame(size_card, bg="white")
        presets_frame.pack(fill="x", pady=(6, 2))
        
        tk.Label(presets_frame, text="Quick Presets:", font=("Segoe UI", 9), bg="white", fg="#555").pack(side="left", padx=(0, 8))
        for p, label in [(50, "50%"), (100, "100%"), (150, "150%"), (200, "200%"), (250, "250%")]:
            btn = tk.Button(presets_frame, text=label, command=lambda val=p: self.set_preset_scale(val), bg="#EAECEE", fg="#333", font=("Segoe UI", 8), relief="groove", padx=6, pady=1, cursor="hand2")
            btn.pack(side="left", padx=2)
            
        btn_fit = tk.Button(presets_frame, text="🎯 Fit Bed (220mm)", command=self.fit_to_bed, bg="#D4EFDF", fg="#0E6251", font=("Segoe UI", 8, "bold"), relief="groove", padx=6, pady=1, cursor="hand2")
        btn_fit.pack(side="left", padx=(6, 2))
        
        btn_reset = tk.Button(presets_frame, text="🔄 Reset", command=lambda: self.set_preset_scale(100), bg="#FADBD8", fg="#78281F", font=("Segoe UI", 8, "bold"), relief="groove", padx=6, pady=1, cursor="hand2")
        btn_reset.pack(side="left", padx=(6, 2))
        
        self.lbl_active_dims = tk.Label(size_card, text="Active Print Size: -", font=("Segoe UI", 9, "bold"), bg="white", fg="#008053")
        self.lbl_active_dims.pack(anchor="w", pady=(6, 0))
        
        # 3. Print Mode Card
        mode_card = tk.LabelFrame(main_frame, text=" 3. Choose Optimization Goal ", font=("Segoe UI", 10, "bold"), bg="white", fg="#2C3E50", padx=12, pady=8)
        mode_card.pack(fill="x", pady=(0, 10))
        
        rb1 = tk.Radiobutton(mode_card, text="🌟 Devotional Idol / Statue / Face (Arachne + Gyroid + Micro Tree Support)", variable=self.mode_var, value="crystal", command=self.update_preview, bg="white", font=("Segoe UI", 9, "bold"), fg="#111", selectcolor="white")
        rb1.pack(anchor="w", pady=2)
        
        rb2 = tk.Radiobutton(mode_card, text="⚙️ Mechanical / Functional (High Strength, 4 Walls, 25% Gyroid, Snap-fit Locks)", variable=self.mode_var, value="mechanical", command=self.update_preview, bg="white", font=("Segoe UI", 9), fg="#333", selectcolor="white")
        rb2.pack(anchor="w", pady=2)
        
        rb3 = tk.Radiobutton(mode_card, text="⚡ Fast Draft (0.20mm Standard, Everyday Fast Prints)", variable=self.mode_var, value="standard", command=self.update_preview, bg="white", font=("Segoe UI", 9), fg="#333", selectcolor="white")
        rb3.pack(anchor="w", pady=2)
        
        # 4. Settings Preview & Cost Estimate Card
        self.settings_card = tk.LabelFrame(main_frame, text=" 4. Auto-Tuned Settings & Cost Estimate ", font=("Segoe UI", 10, "bold"), bg="white", fg="#2C3E50", padx=12, pady=8)
        self.settings_card.pack(fill="x", pady=(0, 10))
        
        self.lbl_preview = tk.Label(self.settings_card, text="• Layer Height: Auto\n• Wall Generator: Arachne\n• Infill: Gyroid (No nozzle scraping)\n• Support: Tree Slim Auto\n• Bed: 65°C Textured PEI | Nozzle: 220°C Bambu PLA", font=("Segoe UI", 9), bg="white", fg="#333", justify="left")
        self.lbl_preview.pack(anchor="w")
        
        self.lbl_estimate = tk.Label(self.settings_card, text="Estimates: -", font=("Segoe UI", 9, "bold"), bg="white", fg="#D35400")
        self.lbl_estimate.pack(anchor="w", pady=(6, 0))
        
        # 5. Action Buttons
        btn_frame = tk.Frame(main_frame, bg="#F4F6F9")
        btn_frame.pack(fill="x", pady=(4, 6))
        
        btn_generate = tk.Button(btn_frame, text="💾 Save Optimized 3MF", command=self.generate_file_only, bg="#2C3E50", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=15, pady=9, cursor="hand2")
        btn_generate.pack(side="left", fill="x", expand=True, padx=(0, 6))
        
        btn_open = tk.Button(btn_frame, text="🚀 Generate & Open in Bambu Studio", command=self.generate_and_open, bg="#008053", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=15, pady=9, cursor="hand2")
        btn_open.pack(side="right", fill="x", expand=True, padx=(6, 0))
        
        # Status Bar
        status_bar = tk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", 9), bg="#EAECEE", fg="#444", anchor="w", padx=15, pady=5)
        status_bar.pack(side="bottom", fill="x")

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select 3D Model",
            filetypes=[("3D Files", "*.stl *.3mf *.obj"), ("3MF files", "*.3mf"), ("STL files", "*.stl"), ("All files", "*.*")]
        )
        if file_path:
            self.input_file.set(file_path)
            self.load_model(file_path)
            
    def load_model(self, file_path):
        try:
            self.status_var.set(f"Analyzing {os.path.basename(file_path)}...")
            self.root.update_idletasks()
            
            self.orig_mesh = load_mesh_safe(file_path)
            self.current_scale = 1.0
            self.scale_pct_var.set("100")
            
            orig_h = float(self.orig_mesh.extents[2])
            self.target_height_var.set(f"{orig_h:.1f}")
            
            w, d, h = float(self.orig_mesh.extents[0]), float(self.orig_mesh.extents[1]), float(self.orig_mesh.extents[2])
            self.lbl_orig_dims.config(text=f"Original Size: Width: {w:.1f} mm  ×  Depth: {d:.1f} mm  ×  Height: {h:.1f} mm")
            
            self.update_preview()
            self.status_var.set("Model loaded! You can adjust size or click Generate.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load 3D model:\n{str(e)}")
            self.status_var.set("Error loading model.")
            
    def apply_target_height(self):
        if not self.orig_mesh:
            return
        try:
            target_h = float(self.target_height_var.get().strip())
            if target_h <= 0:
                raise ValueError("Height must be positive.")
            if target_h > MAX_BED_SIZE:
                messagebox.showwarning("Bed Limit Warning", f"Height ({target_h:.1f} mm) exceeds Bambu A1 bed height (256 mm)!")
            orig_h = float(self.orig_mesh.extents[2])
            self.current_scale = target_h / orig_h
            self.scale_pct_var.set(f"{self.current_scale * 100:.1f}")
            self.update_preview()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid target height number.")

    def apply_scale_pct(self):
        if not self.orig_mesh:
            return
        try:
            pct = float(self.scale_pct_var.get().strip().replace("%", ""))
            if pct <= 0:
                raise ValueError("Scale must be positive.")
            self.current_scale = pct / 100.0
            orig_h = float(self.orig_mesh.extents[2])
            self.target_height_var.set(f"{orig_h * self.current_scale:.1f}")
            self.update_preview()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid scale percentage number.")

    def set_preset_scale(self, pct):
        if not self.orig_mesh:
            return
        self.scale_pct_var.set(str(pct))
        self.apply_scale_pct()

    def fit_to_bed(self):
        if not self.orig_mesh:
            return
        max_extent = max(self.orig_mesh.extents)
        if max_extent > 0:
            scale = 220.0 / max_extent
            self.scale_pct_var.set(f"{scale * 100:.1f}")
            self.apply_scale_pct()

    def get_current_scaled_mesh(self):
        if not self.orig_mesh:
            return None
        m = self.orig_mesh.copy()
        if abs(self.current_scale - 1.0) > 1e-4:
            m.apply_scale(self.current_scale)
        return m

    def update_preview(self):
        scaled_mesh = self.get_current_scaled_mesh()
        if not scaled_mesh:
            return
            
        w, d, h = float(scaled_mesh.extents[0]), float(scaled_mesh.extents[1]), float(scaled_mesh.extents[2])
        self.lbl_active_dims.config(text=f"Active Print Size: Width: {w:.1f} mm  ×  Depth: {d:.1f} mm  ×  Height: {h:.1f} mm  (Scale: {self.current_scale*100:.1f}%)")
        
        _, analysis = analyze_and_optimize_settings(scaled_mesh, self.mode_var.get())
        preview_text = (
            f"• Category: {analysis['category']}\n"
            f"• Preset: {analysis['preset_name']} (Layer Height: {analysis['layer_height']} mm)\n"
            f"• Wall Generator: {analysis['wall_generator']} (Dynamic line width for facial details)\n"
            f"• Outer Wall Speed: {analysis['outer_wall_speed']} (Zero surface vibration)\n"
            f"• Infill Pattern: {analysis['infill']} (Zero nozzle cross-over collisions)\n"
            f"• Wall Loops: {analysis['wall_loops']} perimeters | Support: Tree Slim Auto\n"
            f"• Bed: 65°C Textured PEI Plate | Nozzle: 220°C Bambu PLA Basic"
        )
        self.lbl_preview.config(text=preview_text)
        
        est_text = f"📊 Approx Estimates: Weight: ~{analysis['est_weight_g']:.1f}g  |  Material Cost: ~₹{analysis['est_cost_rs']:.0f}  |  Est Time: ~{analysis['est_time_str']}"
        self.lbl_estimate.config(text=est_text)
        
    def generate_file_only(self):
        output_path = self.process_generation()
        if output_path:
            messagebox.showinfo("Success", f"Optimized 3MF Project successfully created!\n\nLocation:\n{output_path}\n\nYou can now open it in Bambu Studio or send it directly to your Bambu A1!")

    def generate_and_open(self):
        output_path = self.process_generation()
        if output_path:
            if os.path.exists(BAMBU_STUDIO_PATH):
                self.status_var.set("Launching Bambu Studio...")
                subprocess.Popen([BAMBU_STUDIO_PATH, output_path])
                self.status_var.set("Bambu Studio launched with all optimal settings preloaded!")
            else:
                messagebox.showwarning("Notice", f"File created at:\n{output_path}\n\nBambu Studio executable not found at default path, please open the file manually.")

    def process_generation(self):
        path = self.input_file.get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Warning", "Please select a valid 3D model (.stl or .3mf) first.")
            return None
            
        base_name = os.path.splitext(os.path.basename(path))[0]
        dir_name = os.path.dirname(path)
        mode = self.mode_var.get()
        scale_suffix = f"_{int(self.current_scale*100)}pct" if abs(self.current_scale - 1.0) > 1e-4 else ""
        output_path = os.path.join(dir_name, f"{base_name}{scale_suffix}_BambuA1_{mode.capitalize()}_Optimized.3mf")
        
        try:
            self.status_var.set("Generating 1-click optimized 3MF...")
            self.root.update_idletasks()
            generate_bambu_3mf(path, output_path, mode, scale_factor=self.current_scale)
            self.status_var.set(f"Done! Saved: {os.path.basename(output_path)}")
            return output_path
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate optimized 3MF:\n{str(e)}")
            self.status_var.set("Generation failed.")
            return None

if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        model_file = sys.argv[1]
        mode = sys.argv[2] if len(sys.argv) > 2 else "crystal"
        scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        out_file = os.path.splitext(model_file)[0] + f"_BambuA1_{mode.capitalize()}_Optimized.3mf"
        print(f"Generating optimized 3MF for: {model_file} (Mode: {mode}, Scale: {scale})")
        analysis = generate_bambu_3mf(model_file, out_file, mode, scale_factor=scale)
        print("Done! Output:", out_file)
        print("Analysis:", analysis)
        sys.exit(0)
        
    root = tk.Tk()
    app = BambuOptimizerApp(root)
    root.mainloop()
