# 🌳 Improving Data Efficiency of Deep Learning-based Individual Tree Detection and Crown Delineation (ITDCD) in Urban Forests Using Aerial Imagery

This repository contains research conducted at the **University of Canterbury** focusing on **individual tree detection and crown delineation (ITDCD)** in urban forests using aerial RGB imagery.  
The goal of this research is to improve **data efficiency** in deep learning models for ITDCD — achieving strong performance with reduced labeling effort.

🔗 **View the full research content and code explaination here:**  
👉 [https://hartvq37.github.io/geoai/](https://hartvq37.github.io/geoai/)

---

## 📘 Repository Overview

This project presents two major research studies, each available in its own subfolder and as part of the GitHub Pages site:

### 1️⃣ [Improving Individual Tree Detection and Crown Delineation of Urban Trees Using Semi-supervised Deep Learning](tree_detection/)
This study focuses on improving **general ITDCD** by exploring data-efficient approaches:
- Compare supervised learning under varying data sizes  
- Measure dataset informativeness using **entropy metrics**  
- Evaluate **copy-paste augmentation** (data-centric approach)  
- Evaluate **pseudo-label self-training** (model-centric approach)

### 2️⃣ [Urban Individual Tree Species Identification from Aerial Imagery via Deep Active Learning](species_classification/)
This study focuses on **multi-species ITDCD**, aiming to reduce annotation costs:
- Evaluate species classification performance under different data sizes  
- Apply **FixMatch self-training** to leverage unlabelled samples  
- Use **active learning (entropy-based)** to select the most informative samples  
- Propose an integrated **active + self-training** framework for species-level ITDCD

---

## 🧭 Project Summary

Urban forests play a crucial role in environmental sustainability and urban resilience.  
However, large-scale mapping and analysis remain challenging due to:
1. High annotation costs for deep learning training data  
2. Limited exploration of **multi-species** ITDCD approaches  
3. Underutilization of cost-effective **aerial RGB imagery**

This study contributes to developing **data-efficient**, **scalable**, and **accessible** methods for urban forest analysis using deep learning.

---

