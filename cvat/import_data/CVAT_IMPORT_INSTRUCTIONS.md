# CVAT Import Instructions for PlantVillage

## Overview

- **Total Images:** 1565
- **Total Instances:** 1590
- **Healthy Instances:** 456
- **Invalid BBoxes:** 119

## Step-by-Step Import Process

### 1. Start CVAT

```bash
./scripts/setup_cvat.sh
```

### 2. Create Superuser

```bash
docker exec -it cvat python3 manage.py createsuperuser
```

Follow prompts to create username and password.

### 3. Access CVAT

Open browser: http://localhost:8080
Login with your credentials.

### 4. Create Project

1. Click **Projects** → **Create New Project**
2. **Name:** `PlantVillage_Healthy_Reannotation`
3. **Labels:**
   - Add label: `Strawberry___Leaf_scorch` (class 0)
   - Add label: `Strawberry___healthy` (class 1)
4. Click **Submit**

### 5. Create Tasks (One per Split)


#### Task: TRAIN Split

1. Click **Tasks** → **Create New Task**
2. **Name:** `PlantVillage_Train`
3. **Project:** Select `PlantVillage_Healthy_Reannotation`
4. **Select files:**
   - Method: **Share** (mounted directory)
   - Path: `/home/django/share/plantvillage/train/images/`
   - Select all images
5. **Advanced configuration:**
   - Image quality: 95
   - Overlap: 0
   - Segment size: 500
   - Check: **Use zip chunks**
6. Click **Submit**
7. Wait for task creation to complete
8. **Upload annotations** (optional but recommended):
   - Task → Actions → **Upload annotations**
   - Format: **YOLO 1.1**
   - Select labels from: `/home/django/share/plantvillage/{split}/labels/`
   - Click **OK**

   **Stats:** 1095 images, 324 healthy instances, 86 to fix


#### Task: VALID Split

1. Click **Tasks** → **Create New Task**
2. **Name:** `PlantVillage_Valid`
3. **Project:** Select `PlantVillage_Healthy_Reannotation`
4. **Select files:**
   - Method: **Share** (mounted directory)
   - Path: `/home/django/share/plantvillage/valid/images/`
   - Select all images
5. **Advanced configuration:**
   - Image quality: 95
   - Overlap: 0
   - Segment size: 500
   - Check: **Use zip chunks**
6. Click **Submit**
7. Wait for task creation to complete
8. **Upload annotations** (optional but recommended):
   - Task → Actions → **Upload annotations**
   - Format: **YOLO 1.1**
   - Select labels from: `/home/django/share/plantvillage/{split}/labels/`
   - Click **OK**

   **Stats:** 235 images, 77 healthy instances, 18 to fix


#### Task: TEST Split

1. Click **Tasks** → **Create New Task**
2. **Name:** `PlantVillage_Test`
3. **Project:** Select `PlantVillage_Healthy_Reannotation`
4. **Select files:**
   - Method: **Share** (mounted directory)
   - Path: `/home/django/share/plantvillage/test/images/`
   - Select all images
5. **Advanced configuration:**
   - Image quality: 95
   - Overlap: 0
   - Segment size: 500
   - Check: **Use zip chunks**
6. Click **Submit**
7. Wait for task creation to complete
8. **Upload annotations** (optional but recommended):
   - Task → Actions → **Upload annotations**
   - Format: **YOLO 1.1**
   - Select labels from: `/home/django/share/plantvillage/{split}/labels/`
   - Click **OK**

   **Stats:** 235 images, 55 healthy instances, 15 to fix

### 6. Start Annotating

#### Priority 1: Fix Invalid BBoxes

Use the priority lists to focus on images that need fixing:

- **Train:** See `train_priority_images.txt` (86 images)
- **Valid:** See `valid_priority_images.txt` (18 images)
- **Test:** See `test_priority_images.txt` (15 images)

#### Priority 2: Review All Healthy Samples

After fixing invalid bboxes, review all healthy instances for quality.

#### Priority 3: Full Dataset (Optional)

If time permits, review entire dataset for missed annotations.

### 7. Export Annotations

When annotation is complete:

1. Go to each Task
2. Actions → **Export annotations**
3. Format: **YOLO 1.1**
4. Download ZIP
5. Extract to: `data/external/plantvillage_healthy_reannotated/{split}/`

### 8. Validate

```bash
python scripts/validate_cvat_export.py \
  --input data/external/plantvillage_healthy_reannotated/ \
  --original data/external/plantvillage_strawberry/
```

## Annotation Guidelines

See detailed guidelines: `docs/CVAT_ANNOTATION_GUIDELINES.md`

## Troubleshooting

**CVAT not starting?**
```bash
docker compose -f docker-compose.cvat.yml down
docker compose -f docker-compose.cvat.yml up -d
```

**Can't see PlantVillage images?**
```bash
docker exec -it cvat ls -la /home/django/share/plantvillage/
```

**View CVAT logs:**
```bash
docker logs cvat -f
```

