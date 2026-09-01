import os
import glob
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

BASE_PATH = (
    r"D:\programs\LearningFundamentals\Ml_clg\project_ml\CODD"
    r"\Construction and Demolition Waste Object Detection Dataset  (CODD)"
)

SPLITS = [
    "training",
    "validation",
    "testing"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_unique_filename(base_name):
    if not os.path.exists(base_name):
        return base_name

    name, ext = os.path.splitext(base_name)
    counter = 1

    while os.path.exists(f"{name}_{counter}{ext}"):
        counter += 1

    return f"{name}_{counter}{ext}"


def assign_subtype(material):
    material = str(material).lower()

    if "wood" in material:
        return "Treated"

    if "concrete" in material:
        return "Reinforced"

    if "metal" in material:
        return "Steel"

    return "Standard"


# ============================================================
# PROCESS DATASET
# ============================================================

for split in SPLITS:

    folder_path = os.path.join(
        BASE_PATH,
        split
    )

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    xml_files = glob.glob(
        os.path.join(
            folder_path,
            "*.xml"
        )
    )

    data = []

    parsed_xml_count = len(xml_files)
    objects_converted = 0
    sample_counter = 1

    # ========================================================
    # READ XML FILES
    # ========================================================

    for xml_file in xml_files:

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # ------------------------------------------------
            # IMAGE NAME
            # ------------------------------------------------

            img_name_node = root.find("filename")

            if img_name_node is not None:
                img_filename = img_name_node.text
            else:
                img_filename = (
                    os.path.basename(xml_file)
                    .replace(".xml", ".jpg")
                )

            img_path = os.path.join(
                folder_path,
                img_filename
            )

            # ------------------------------------------------
            # FALLBACK IMAGE PATHS
            # ------------------------------------------------

            if not os.path.exists(img_path):

                img_path = (
                    xml_file
                    .replace(".xml", ".jpg")
                )

            if not os.path.exists(img_path):

                img_path = (
                    xml_file
                    .replace(".xml", ".JPG")
                )

            if not os.path.exists(img_path):
                continue

            # ------------------------------------------------
            # READ IMAGE
            # ------------------------------------------------

            img = cv2.imread(img_path)

            if img is None:
                continue

            img_h, img_w = img.shape[:2]

            # =================================================
            # PROCESS EVERY OBJECT IN XML
            # =================================================

            for obj in root.findall("object"):

                try:

                    # -----------------------------------------
                    # OBJECT CLASS
                    # -----------------------------------------

                    obj_class = (
                        obj.find("name").text
                    )

                    # -----------------------------------------
                    # BOUNDING BOX
                    # -----------------------------------------

                    bndbox = obj.find("bndbox")

                    xmin_val = int(
                        float(
                            bndbox.find(
                                "xmin"
                            ).text
                        )
                    )

                    ymin_val = int(
                        float(
                            bndbox.find(
                                "ymin"
                            ).text
                        )
                    )

                    xmax_val = int(
                        float(
                            bndbox.find(
                                "xmax"
                            ).text
                        )
                    )

                    ymax_val = int(
                        float(
                            bndbox.find(
                                "ymax"
                            ).text
                        )
                    )

                    # -----------------------------------------
                    # CLAMP BOUNDING BOX TO IMAGE
                    # -----------------------------------------

                    xmin = max(
                        0,
                        xmin_val
                    )

                    ymin = max(
                        0,
                        ymin_val
                    )

                    xmax = min(
                        img_w,
                        xmax_val
                    )

                    ymax = min(
                        img_h,
                        ymax_val
                    )

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    # -----------------------------------------
                    # CROP OBJECT
                    # -----------------------------------------

                    crop = img[
                        ymin:ymax,
                        xmin:xmax
                    ]

                    obj_h, obj_w = crop.shape[:2]

                    if obj_h == 0 or obj_w == 0:
                        continue

                    # -----------------------------------------
                    # BASIC OBJECT MEASUREMENTS
                    # -----------------------------------------

                    obj_area = (
                        obj_w * obj_h
                    )

                    aspect_ratio = (
                        obj_w / obj_h
                    )

                    relative_area = (
                        obj_area /
                        (img_w * img_h)
                    )

                    # =================================================
                    # GRAYSCALE FEATURES
                    # =================================================

                    gray = cv2.cvtColor(
                        crop,
                        cv2.COLOR_BGR2GRAY
                    )

                    mean_brightness = np.mean(
                        gray
                    )

                    std_brightness = np.std(
                        gray
                    )

                    contrast = (
                        int(np.max(gray))
                        -
                        int(np.min(gray))
                    )

                    # =================================================
                    # COLOR FEATURES
                    # =================================================

                    b, g, r = cv2.split(crop)

                    mean_b = np.mean(b)
                    mean_g = np.mean(g)
                    mean_r = np.mean(r)

                    color_std = np.std(
                        crop
                    )

                    # =================================================
                    # HSV FEATURES
                    # =================================================

                    hsv = cv2.cvtColor(
                        crop,
                        cv2.COLOR_BGR2HSV
                    )

                    mean_hue = np.mean(
                        hsv[:, :, 0]
                    )

                    mean_sat = np.mean(
                        hsv[:, :, 1]
                    )

                    std_sat = np.std(
                        hsv[:, :, 1]
                    )

                    # =================================================
                    # EDGE FEATURES
                    # =================================================

                    edges = cv2.Canny(
                        gray,
                        100,
                        200
                    )

                    edge_density = (
                        np.sum(edges > 0)
                        /
                        obj_area
                    )

                    # =================================================
                    # TEXTURE
                    # =================================================

                    texture_laplacian_var = (
                        cv2.Laplacian(
                            gray,
                            cv2.CV_64F
                        ).var()
                    )

                    # =================================================
                    # DARK / BRIGHT PIXELS
                    # =================================================

                    dark_pixel_ratio = (
                        np.sum(gray < 50)
                        /
                        obj_area
                    )

                    bright_pixel_ratio = (
                        np.sum(gray > 205)
                        /
                        obj_area
                    )

                    # =================================================
                    # OBJECT CONTOUR
                    # =================================================

                    _, thresh = cv2.threshold(
                        gray,
                        0,
                        255,
                        cv2.THRESH_BINARY
                        +
                        cv2.THRESH_OTSU
                    )

                    contours, _ = cv2.findContours(
                        thresh,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    contour_area = 0
                    contour_perimeter = 0
                    fill_ratio = 0
                    circularity = 0
                    solidity = 0
                    extent = 0

                    if contours:

                        largest_contour = max(
                            contours,
                            key=cv2.contourArea
                        )

                        # -----------------------------
                        # CONTOUR AREA
                        # -----------------------------

                        contour_area = cv2.contourArea(
                            largest_contour
                        )

                        # -----------------------------
                        # PERIMETER
                        # -----------------------------

                        contour_perimeter = cv2.arcLength(
                            largest_contour,
                            True
                        )

                        # -----------------------------
                        # FILL RATIO
                        # -----------------------------

                        fill_ratio = (
                            contour_area
                            /
                            obj_area
                        )

                        # -----------------------------
                        # CIRCULARITY
                        # -----------------------------

                        if contour_perimeter > 0:

                            circularity = (
                                4
                                *
                                np.pi
                                *
                                contour_area
                                /
                                (
                                    contour_perimeter
                                    ** 2
                                )
                            )

                        # -----------------------------
                        # SOLIDITY
                        # -----------------------------

                        hull = cv2.convexHull(
                            largest_contour
                        )

                        hull_area = cv2.contourArea(
                            hull
                        )

                        if hull_area > 0:

                            solidity = (
                                contour_area
                                /
                                hull_area
                            )

                        # -----------------------------
                        # EXTENT
                        # -----------------------------

                        x, y, w, h = cv2.boundingRect(
                            largest_contour
                        )

                        bounding_area = (
                            w * h
                        )

                        if bounding_area > 0:

                            extent = (
                                contour_area
                                /
                                bounding_area
                            )

                    # =================================================
                    # APPEND IMAGE-DERIVED FEATURES
                    # =================================================

                    data.append({

                        "Sample_ID":
                            f"{split.upper()}_{sample_counter:05d}",

                        "Image_ID":
                            os.path.basename(
                                img_path
                            ),

                        "Material_Type":
                            obj_class,

                        "pixel_img_width":
                            img_w,

                        "pixel_img_height":
                            img_h,

                        "pixel_obj_width":
                            obj_w,

                        "pixel_obj_height":
                            obj_h,

                        "relative_area":
                            relative_area,

                        "edge_density":
                            edge_density,

                        "fill_ratio":
                            fill_ratio,

                        "color_std":
                            color_std,

                        "mean_brightness":
                            mean_brightness,

                        "std_brightness":
                            std_brightness,

                        "contrast":
                            contrast,

                        "texture_laplacian_var":
                            texture_laplacian_var,

                        "dark_pixel_ratio":
                            dark_pixel_ratio,

                        "bright_pixel_ratio":
                            bright_pixel_ratio,

                        "mean_hue":
                            mean_hue,

                        "mean_saturation":
                            mean_sat,

                        "saturation_std":
                            std_sat,

                        "contour_area":
                            contour_area,

                        "contour_perimeter":
                            contour_perimeter,

                        "circularity":
                            circularity,

                        "solidity":
                            solidity,

                        "extent":
                            extent

                    })

                    objects_converted += 1
                    sample_counter += 1

                except Exception as e:

                    print(
                        f"Object error in "
                        f"{xml_file}: {e}"
                    )

        except Exception as e:

            print(
                f"XML error in "
                f"{xml_file}: {e}"
            )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    if not data:
        print(
            f"No data found for {split}"
        )
        continue

    df = pd.DataFrame(data)

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    df["Source"] = (
        "Demolition_Site"
    )

    df["Exposure_Type"] = (
        "Exterior_Weathered"
    )

    df["Material_Subtype"] = (
        df["Material_Type"]
        .apply(assign_subtype)
    )

    # ========================================================
    # DAMAGE FEATURES
    # ========================================================

    # --------------------------------------------------------
    # CRACK LEVEL
    # --------------------------------------------------------
    #
    # This is an IMAGE-BASED EDGE INDICATOR.
    # It should not be described as a direct measurement
    # of physical cracks.
    #
    # More visible edges/details -> higher level.

    df["Crack_Level"] = np.clip(
        (
            df["edge_density"] * 100
        ).astype(int),
        1,
        5
    )

    # --------------------------------------------------------
    # BREAKAGE LEVEL
    # --------------------------------------------------------
    #
    # Lower fill ratio means the detected contour occupies
    # less of its bounding region.

    df["Breakage_Level"] = np.clip(
        (
            5 -
            (
                df["fill_ratio"] * 5
            ).astype(int)
        ),
        1,
        5
    )

    # --------------------------------------------------------
    # MISSING PORTION
    # --------------------------------------------------------

    df["Missing_Portion"] = np.where(

        df["fill_ratio"] < 0.4,
        "Severe",

        np.where(
            df["fill_ratio"] < 0.6,
            "High",

            np.where(
                df["fill_ratio"] < 0.8,
                "Medium",
                "Low"
            )
        )
    )

    # --------------------------------------------------------
    # DISCOLORATION LEVEL
    # --------------------------------------------------------
    #
    # Based on measured color variation.
    # This indicates color non-uniformity, not proven
    # physical discoloration.

    df["Discoloration_Level"] = np.where(

        df["color_std"] > 60,
        "High",

        np.where(
            df["color_std"] > 30,
            "Moderate",
            "Low"
        )
    )

    # --------------------------------------------------------
    # DEFORMATION LEVEL
    # --------------------------------------------------------
    #
    # Based on the object's bounding-box aspect ratio.

    aspect_ratio = (
        df["pixel_obj_width"]
        /
        df["pixel_obj_height"]
    )

    deviation = np.abs(
        np.log(
            aspect_ratio
        )
    )

    df["Deformation_Level"] = np.select(

        [
            deviation < 0.15,
            deviation < 0.30,
            deviation < 0.50,
            deviation < 0.80
        ],

        [
            1,
            2,
            3,
            4
        ],

        default=5
    )

    # ========================================================
    # SURFACE DAMAGE
    # ========================================================
    #
    # Derived from the existing image measurements.
    #
    # Surface damage uses:
    #   - edge density
    #   - texture variation
    #   - color variation
    #   - contrast
    #
    # Each measurement is converted to a 1-5 level using
    # the existing style of threshold-based scoring.

    surface_texture_level = np.clip(
        (
            df["texture_laplacian_var"]
            /
            1000
        ).astype(int) + 1,
        1,
        5
    )

    surface_edge_level = np.clip(
        (
            df["edge_density"] * 100
        ).astype(int),
        1,
        5
    )

    surface_color_level = np.select(

        [
            df["color_std"] > 60,
            df["color_std"] > 30
        ],

        [
            5,
            3
        ],

        default=1
    )

    surface_contrast_level = np.clip(
        (
            df["contrast"] / 50
        ).astype(int) + 1,
        1,
        5
    )

    df["Surface_Damage"] = (

        (
            surface_texture_level
            +
            surface_edge_level
            +
            surface_color_level
            +
            surface_contrast_level
        )
        /
        4

    ).round(1)

    # ========================================================
    # OVERALL DAMAGE SCORE
    # ========================================================
    #
    # Uses the three existing image-derived indicators:
    #
    #   Crack_Level
    #   Breakage_Level
    #   Deformation_Level
    #
    # Higher = more visible damage.

    df["Damage_Score"] = (

        (
            df["Crack_Level"]
            +
            df["Breakage_Level"]
            +
            df["Deformation_Level"]
        )
        /
        3

    ).round(1)

    # ========================================================
    # CONDITION SCORE
    # ========================================================
    #
    # Higher damage -> lower condition.

    df["Condition_Score"] = np.clip(

        (
            10
            -
            (
                df["Damage_Score"] * 1.5
            )
        ).round(1),

        1,
        10
    )

    # ========================================================
    # CONDITION CLASS
    # ========================================================

    df["Condition_Class"] = np.where(

        df["Condition_Score"] >= 8,
        "Excellent",

        np.where(
            df["Condition_Score"] >= 6,
            "Good",

            np.where(
                df["Condition_Score"] >= 4,
                "Fair",
                "Poor"
            )
        )
    )

    # ========================================================
    # STRUCTURAL INTEGRITY
    # ========================================================

    df["Structural_Integrity"] = np.where(

        df["Condition_Score"] >= 5,

        "Visually Intact",

        "Visually Compromised"
    )

    # ========================================================
    # RECOVERY PATHWAY
    # ========================================================

    df["Recovery_Pathway"] = np.where(

        df["Condition_Score"] >= 7,

        "Direct Reuse",

        np.where(

            df["Condition_Score"] >= 4,

            "Refurbishment / Alternative Use",

            "Alternative Use / Recycling"
        )
    )

    # ========================================================
    # RECONDITIONING REQUIRED
    # ========================================================

    df["Reconditioning_Required"] = np.where(

        df["Condition_Score"] < 7,

        "Yes",

        "No"
    )

    # ========================================================
    # RECOVERY POTENTIAL
    # ========================================================

    df["Recovery_Potential"] = np.where(

        df["Condition_Score"] >= 7,

        "High",

        np.where(

            df["Condition_Score"] >= 4,

            "Medium",

            "Low"
        )
    )

    # ========================================================
    # ALTERNATIVE USE
    # ========================================================

    def alternative_use(
        material,
        score
    ):

        material = str(
            material
        ).lower()

        if score >= 7:
            return (
                "Direct Construction Reuse"
            )

        if score >= 4:

            if "brick" in material:
                return (
                    "Landscaping / Garden Edging"
                )

            if "tile" in material:
                return (
                    "Mosaic / Decorative Use"
                )

            if "wood" in material:
                return (
                    "Furniture / Decorative Use"
                )

            if "concrete" in material:
                return (
                    "Landscaping Use"
                )

            return (
                "Secondary Non-Construction Use"
            )

        if (
            "brick" in material
            or
            "concrete" in material
            or
            "tile" in material
        ):

            return "Aggregate / Filler"

        if "wood" in material:
            return "Wood Recycling"

        if "plastic" in material:
            return "Plastic Recycling"

        return "Material Recycling"

    df["Alternative_Use_Category"] = df.apply(

        lambda row:
        alternative_use(
            row["Material_Type"],
            row["Condition_Score"]
        ),

        axis=1
    )

    # ========================================================
    # ALTERNATIVE USE SUITABILITY
    # ========================================================

    df["Alternative_Use_Suitability"] = (

        df["Condition_Score"] * 10

    ).astype(int)

    # ========================================================
    # FINAL COLUMN ORDER
    # ========================================================

    columns_order = [

        # Identification
        "Sample_ID",
        "Image_ID",
        "Material_Type",
        "Material_Subtype",

        # Dataset context
        "Source",
        "Exposure_Type",

        # Damage-related image measurements
        "Crack_Level",
        "Surface_Damage",
        "Breakage_Level",
        "Discoloration_Level",
        "Deformation_Level",
        "Missing_Portion",

        # Damage / condition
        "Damage_Score",
        "Condition_Score",
        "Condition_Class",

        # Recovery
        "Structural_Integrity",
        "Recovery_Pathway",
        "Reconditioning_Required",
        "Recovery_Potential",
        "Alternative_Use_Category",
        "Alternative_Use_Suitability",

        # Image geometry
        "pixel_img_width",
        "pixel_img_height",
        "pixel_obj_width",
        "pixel_obj_height",
        "relative_area",

        # RGB / texture features
        "edge_density",
        "fill_ratio",
        "color_std",
        "mean_brightness",
        "std_brightness",
        "contrast",
        "texture_laplacian_var",
        "dark_pixel_ratio",
        "bright_pixel_ratio",
        "mean_hue",
        "mean_saturation",
        "saturation_std",

        # Shape features
        "contour_area",
        "contour_perimeter",
        "circularity",
        "solidity",
        "extent"
    ]

    df = df[
        columns_order
    ]

    # ========================================================
    # SAVE CSV
    # ========================================================

    out_filename = (
        f"{split}_image_features.csv"
    )

    out_path = get_unique_filename(
        out_filename
    )

    df.to_csv(
        out_path,
        index=False
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 60)

    print(
        f"Folder Path: {folder_path}"
    )

    print(
        f"XML Files Found: {parsed_xml_count}"
    )

    print(
        f"Objects Converted: {objects_converted}"
    )

    print(
        f"Number of Rows: {len(df)}"
    )

    print(
        f"Number of Columns: {len(df.columns)}"
    )

    print(
        f"Output CSV: {os.path.abspath(out_path)}"
    )

    print("=" * 60)