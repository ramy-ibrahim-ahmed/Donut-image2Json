<<<<<<< HEAD
# 🧾 Donut Model Fine-Tuning for Receipt Parsing

![Donut Model Architecture](donut_architecture.jpg)
*Visual Overview of Donut Model Architecture on its paper ([Source](https://arxiv.org/abs/2111.15664))*

A powerful implementation of the Donut (Document Understanding Transformer) model fine-tuned for extracting structured JSON data from receipt images. This project leverages the ICDAR-2019-SROIE dataset to create a robust document understanding solution.

## 🌟 Features

- **Advanced Document Understanding**: Built on naver-clova-ix/donut-base
- **Efficient Processing**: Optimized for 720x960 receipt images
- **Smart Token Handling**: Dynamic special token generation
- **Training Optimization**:
  - Gradient Accumulation (32 steps)
  - Cosine Learning Rate Scheduler
  - Mixed Precision Training (FP16)
  - Early Stopping

## 📊 Training Metrics
![Training Metrics Plot](output.png)

=======
# ReceiptVision: Donut-Powered Document Intelligence

## Project Description
ReceiptVision is a streamlined AI system that fine-tunes the Donut Vision Transformer (ViT) to automate the conversion of receipts into structured JSON data. By focusing on real-world usability, the project eliminates the inefficiencies of manual data entry, delivering precise extraction of transaction details (e.g., dates, totals, vendor names) even from low-quality scans or handwritten notes.

---

## Why It Stands Out:

- OCR-Free Design: Leverages Donut’s end-to-end document understanding, bypassing traditional OCR error-prone pipelines.
- Domain Adaptability: Easily extendable to invoices, tickets, or forms with minimal retraining.
- Developer-Friendly: Includes a lightweight API wrapper for integration into apps, workflows, or cloud services.
>>>>>>> 357e07886c3cc123c1918f075e4d703a7808737e
