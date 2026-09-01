import os
import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pandas as pd

def get_unique_filename(base_name):
    if not os.path.exists(base_name):
        return base_name
    name, ext = os.path.splitext(base_name)
    counter = 1
    while os.path.exists(f"{name}_{counter}{ext}"):
        counter += 1
    return f"{name}_{counter}{ext}"

def assign_subtype(mat):
    mat = str(mat).lower()
    if 'wood' in mat: return 'Treated'
    if 'concrete' in mat: return 'Reinforced'
    if 'metal' in mat: return 'Steel'
    return 'Standard'

base_path = r"D:\programs\LearningFundamentals\Ml_clg\project_ml\CODD\Construction and Demolition Waste Object Detection Dataset  (CODD)"
splits = ['training', 'validation', 'testing']

for split in splits:
    folder_path = os.path.join(base_path, split)
    if not os.path.exists(folder_path):
        continue
        
    xml_files = glob.glob(os.path.join(folder_path, "*.xml"))
    
    data = []
    parsed_xml_count = len(xml_files)
    objects_converted = 0
    sample_counter = 1
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            img_name_node = root.find("filename")
            if img_name_node is not None:
                img_filename = img_name_node.text
            else:
                img_filename = os.path.basename(xml_file).replace('.xml', '.jpg')
                
            img_path = os.path.join(folder_path, img_filename)
            
            if not os.path.exists(img_path):
                img_path = xml_file.replace('.xml', '.jpg')
                if not os.path.exists(img_path):
                    img_path = xml_file.replace('.xml', '.JPG')
                    
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img_h, img_w = img.shape[:2]
            
            for obj in root.findall("object"):
                try:
                    obj_class = obj.find("name").text
                    bndbox = obj.find("bndbox")
                    
                    xmin_val = int(float(bndbox.find("xmin").text))
                    ymin_val = int(float(bndbox.find("ymin").text))
                    xmax_val = int(float(bndbox.find("xmax").text))
                    ymax_val = int(float(bndbox.find("ymax").text))
                    
                    xmin = max(0, xmin_val)
                    ymin = max(0, ymin_val)
                    xmax = min(img_w, xmax_val)
                    ymax = min(img_h, ymax_val)
                    
                    if xmax <= xmin or ymax <= ymin:
                        continue
                        
                    crop = img[ymin:ymax, xmin:xmax]
                    obj_h, obj_w = crop.shape[:2]
                    
                    if obj_h == 0 or obj_w == 0:
                        continue
                        
                    obj_area = obj_w * obj_h
                    aspect_ratio = obj_w / obj_h if obj_h > 0 else 0
                    relative_area = obj_area / (img_w * img_h) if (img_w * img_h) > 0 else 0
                    
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    mean_brightness = np.mean(gray)
                    std_brightness = np.std(gray)
                    contrast = int(np.max(gray)) - int(np.min(gray))
                    
                    b, g, r = cv2.split(crop)
                    mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
                    color_std = np.std(crop)
                    
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    mean_hue = np.mean(hsv[:, :, 0])
                    mean_sat = np.mean(hsv[:, :, 1])
                    
                    edges = cv2.Canny(gray, 100, 200)
                    edge_density = np.sum(edges > 0) / obj_area if obj_area > 0 else 0
                    
                    texture = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    dark_pixel_ratio = np.sum(gray < 50) / obj_area if obj_area > 0 else 0
                    bright_pixel_ratio = np.sum(gray > 205) / obj_area if obj_area > 0 else 0
                    
                    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    contour_area = 0
                    contour_perimeter = 0
                    fill_ratio = 0
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        contour_area = cv2.contourArea(largest_contour)
                        contour_perimeter = cv2.arcLength(largest_contour, True)
                        fill_ratio = contour_area / obj_area if obj_area > 0 else 0
                        
                    data.append({
                        'Sample_ID': f"{split.upper()}_{sample_counter:05d}",
                        'Image_ID': os.path.basename(img_path),
                        'Material_Type': obj_class,
                        'pixel_img_width': img_w,
                        'pixel_img_height': img_h,
                        'pixel_obj_width': obj_w,
                        'pixel_obj_height': obj_h,
                        'edge_density': edge_density,
                        'fill_ratio': fill_ratio,
                        'color_std': color_std,
                        'mean_brightness': mean_brightness,
                        'contrast': contrast,
                        'texture_laplacian_var': texture
                    })
                    objects_converted += 1
                    sample_counter += 1
                    
                except Exception:
                    pass
        except Exception:
            pass
            
    if data:
        df = pd.DataFrame(data)
        df['Length_mm'] = np.nan
        df['Width_mm'] = np.nan
        df['Thickness_mm'] = np.nan
        df['Source'] = 'Demolition_Site'
        df['Exposure_Type'] = 'Exterior_Weathered'
        df['Material_Subtype'] = df['Material_Type'].apply(assign_subtype)
        
        df['Crack_Level'] = np.clip((df['edge_density'] * 100).astype(int), 1, 5)
        df['Breakage_Level'] = np.clip(5 - (df['fill_ratio'] * 5).astype(int), 1, 5)
        
        df['Missing_Portion'] = np.where(df['fill_ratio'] < 0.4, 'Severe',
                                np.where(df['fill_ratio'] < 0.6, 'High',
                                np.where(df['fill_ratio'] < 0.8, 'Medium', 'Low')))
                                
        df['Discoloration_Level'] = np.where(df['color_std'] > 60, 'High',
                                    np.where(df['color_std'] > 30, 'Moderate', 'Low'))
                                    
        aspect_ratio = (
            df['pixel_obj_width'] /
            df['pixel_obj_height']
        )

        deviation = abs(
            np.log(aspect_ratio)
        )

        df['Deformation_Level'] = np.select(
            [
                deviation < 0.15,
                deviation < 0.30,
                deviation < 0.50,
                deviation < 0.80
            ],
            [1, 2, 3, 4],
            default=5
        )
        
        df['Moisture_Dampness_Level'] = np.where(df['mean_brightness'] < 80, 'Wet',
                                        np.where(df['mean_brightness'] < 120, 'Damp', 'Dry'))
                                        
        df['Surface_Contamination_Level'] = np.clip((df['texture_laplacian_var'] / 1000).astype(int), 1, 5)
        
        df['Damage_Score'] = ((df['Crack_Level'] + df['Breakage_Level'] + df['Deformation_Level']) / 3).round(1)
        df['Condition_Score'] = np.clip((10 - df['Damage_Score'] * 1.5).round(1), 1, 10)
        
        df['Condition_Class'] = np.where(df['Condition_Score'] >= 8, 'Excellent',
                                np.where(df['Condition_Score'] >= 6, 'Good',
                                np.where(df['Condition_Score'] >= 4, 'Fair', 'Poor')))
                                
        df['Structural_Integrity'] = np.where(df['Condition_Score'] >= 5, 'Intact', 'Compromised')
        
        df['Recovery_Pathway'] = np.where(
            df['Condition_Score'] >= 7,
            'Direct Reuse',
            np.where(
                df['Condition_Score'] >= 4,
                'Refurbishment / Alternative Use',
                'Alternative Use / Recycling'
            )
        )
        def alternative_use(material, score):
            material = str(material).lower()

            if score >= 7:
                return 'Direct Construction Reuse'

            if score >= 4:
                if 'brick' in material:
                    return 'Landscaping / Garden Edging'
                elif 'tile' in material:
                    return 'Mosaic / Decorative Use'
                elif 'wood' in material:
                    return 'Furniture / Decorative Use'
                elif 'concrete' in material:
                    return 'Landscaping Use'
                else:
                    return 'Secondary Non-Construction Use'

            if 'brick' in material or 'concrete' in material or 'tile' in material:
                return 'Aggregate / Filler'
            elif 'wood' in material:
                return 'Wood Recycling'
            elif 'plastic' in material:
                return 'Plastic Recycling'
            else:
                return 'Material Recycling'


        df['Alternative_Use_Category'] = df.apply(
            lambda row: alternative_use(
                row['Material_Type'],
                row['Condition_Score']
            ),
            axis=1
        )
                                 
        df['Reconditioning_Required'] = np.where(df['Recovery_Pathway'] == 'Refurbishment', 'Yes', 'No')
        
        df['Recovery_Potential'] = np.where(df['Condition_Score'] >= 7, 'High',
                                   np.where(df['Condition_Score'] >= 4, 'Medium', 'Low'))
                                   
        df['Alternative_Use_Category'] = np.where(df['Condition_Score'] < 5, 'Aggregate/Filler', 'Architectural/Structural')
        df['Alternative_Use_Suitability'] = (df['Condition_Score'] * 10).astype(int)

        columns_order = [
            'Sample_ID', 'Image_ID', 'Material_Type', 'Material_Subtype', 'Source', 
            'Exposure_Type', 'Length_mm', 'Width_mm', 'Thickness_mm', 'Crack_Level', 
            'Surface_Damage', 'Breakage_Level', 'Discoloration_Level', 'Deformation_Level', 
            'Missing_Portion', 'Moisture_Dampness_Level', 'Surface_Contamination_Level', 
            'Damage_Score', 'Condition_Score', 'Condition_Class', 'Structural_Integrity', 
            'Recovery_Pathway', 'Reconditioning_Required', 'Recovery_Potential', 
            'Alternative_Use_Category', 'Alternative_Use_Suitability',
            'pixel_img_width', 'pixel_img_height', 'pixel_obj_width', 'pixel_obj_height',
            'edge_density', 'fill_ratio', 'color_std', 'mean_brightness', 'contrast', 'texture_laplacian_var'
        ]
        
        df['Surface_Damage'] = df['Damage_Score'] 
        
        df = df[columns_order]
        
        out_filename = f"{split}_image_features.csv"
        out_path = get_unique_filename(out_filename)
        df.to_csv(out_path, index=False)
        
        print(f"Folder Path: {folder_path}")
        print(f"XML Files Found: {parsed_xml_count}")
        print(f"Objects Converted: {objects_converted}")
        print(f"Number of Columns: {len(df.columns)}")
        print(f"Output CSV Path: {os.path.abspath(out_path)}")
        print("-" * 50)