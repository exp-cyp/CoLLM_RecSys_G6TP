# คู่มือทำความเข้าใจ CoLLM และเตรียมสอบ

เริ่มอ่าน `data.py` → `models.py` → `train.py` → `checkpoints.py` → `evaluate.py`
ใช้ข้อมูลหนึ่งแถวตามไปจนถึงคะแนนทำนาย ก่อนขยายไปดู training loop ทั้งชุด

## เส้นทางข้อมูลหนึ่งตัวอย่าง

| ขั้น | ข้อมูล / shape เมื่อ batch = 4 |
|---|---|
| JSONL record | uid, iid, title, history_titles, timestamps และ label |
| PromptDataset / Collator | input_ids และ attention_mask `[4, L]` พร้อม feature positions |
| MF lookup | user vector และ item vector อย่างละ `[4, 64]` |
| Shared mapper | 64 → 640 → 896; ได้ features อย่างละ `[4, 896]` |
| Hybrid input | text embeddings ผสม mapped features `[4, L, 896]` |
| Qwen decoder | เลือก final hidden state `[4, 896]` |
| Output head | คำนวณเฉพาะแถว Yes ได้ raw scores `[4]` |
| Loss / evaluation | BCE กับ labels หรือ AUC/ranking จาก continuous scores |

`L` คือความยาว prompt หลัง padding ใน batch นั้น สูงสุด 256 ตาม config ที่เสนอ
ไม่มี `history_ids` ใน schema ปัจจุบันเพราะโมเดลใช้ชื่อหนังใน history ส่วน MF ใช้ uid/iid ปัจจุบัน
Labels อยู่ใน batch สำหรับ loss แต่แยกออกก่อนเรียก `CoLLM.forward()`

## กลไกที่ทุกคนต้องอธิบายได้

1. **MF:** เรียน user/item vectors แล้วใช้ dot product ทำนายความชอบด้วย BCE
   แต่ละมิติเป็น latent feature ไม่ได้กำหนดว่าเป็น genre ใดล่วงหน้า
2. **TALLRec-style:** ใช้ชื่อหนังในประวัติและชื่อหนังเป้าหมายเป็นข้อความ ฝึกเฉพาะ LoRA
3. **CIE:** ใช้ MF ดึง collaborative vectors แล้วผ่าน shared MLP ให้มีมิติตรงกับ Qwen
   จากนั้นแทน input embeddings ที่ตำแหน่ง user/item placeholders
4. **สอง stages:** Stage 1 ฝึก LoRA จากข้อความ; Stage 2 ตรึง LoRA/MF/Qwen แล้วฝึก MLP
5. **Freeze กับ no_grad:** freeze ป้องกันการอัปเดตน้ำหนัก แต่ Stage 2 ยังต้องส่ง gradient
   ผ่าน frozen Qwen กลับไปหา MLP จึงใช้ `no_grad()` ครอบ decoder ระหว่างฝึกไม่ได้
6. **Scoring:** ใช้ raw Yes logit ตาม upstream Qwen BCE; sigmoid แปลเป็น binary model
   probability ได้ แต่ไม่ใช่ vocabulary-softmax probability หรือ softmax ระหว่าง Yes/No
7. **การประหยัด memory:** decoder ยังประมวลผลทั้ง prompt แต่ output projection
   คำนวณเฉพาะหนึ่งแถวของ Yes ไม่สร้าง logits ของ vocabulary ทุกคำในทุกตำแหน่ง
8. **Unseen IDs:** MF features เป็นศูนย์ก่อนเข้า mapper; bias ของ MLP อาจให้ mapped
   feature ที่เป็นค่าคงที่และไม่เป็นศูนย์ จึงไม่ใช่ random embedding ที่ยังไม่เคยฝึก

## ข้อมูล, leakage และ evaluation

- Rating `> 3` เป็น positive; ที่เหลือเป็น negative จาก observed ratings จริง
- แบ่ง train/validation/test ตามเวลา ก่อนเลือก subsets สำหรับการทดลอง
- History ใช้เฉพาะ positive feedback ก่อนเป้าหมาย ไม่เกิน 10 เรื่อง
  เหตุการณ์ timestamp เท่ากันไม่เห็นคำตอบของกันและกัน
- Rolling observed-feedback protocol อนุญาตให้ held-out feedback ที่เกิดก่อนหน้า
  เข้า history ของเหตุการณ์ถัดไป แต่ไม่ใช้ held-out labels ปรับ parameters
- Warm หมายถึงทั้ง user และ item มีอย่างน้อย 3 interactions ใน training selection
- AUC วัดความสามารถให้ positive ได้คะแนนสูงกว่า negative ไม่ใช่ accuracy
  Warm/Cold AUC เป็น metric เดียวกันที่แบ่งกลุ่ม ไม่ใช่ UAUC รายผู้ใช้
- Ranking ตัวอย่างตรึง user/history/เวลา แล้วเรียง candidate scores โดยไม่ใช้ labels
  เป็นตัวอย่างการใช้ prediction ไม่ใช่ full-catalog top-k evaluation

## Training loop และ checkpoint identity

- Batch 4 และ accumulation 2 รวม gradient ก่อน update หนึ่งครั้งเท่ากับ effective batch 8
- `max_steps=512` หมายถึง optimizer updates ไม่ใช่ epochs
- ใช้ validation AUC เลือก checkpoints หลังมี optimizer update แล้ว ไม่ใช้ test scores
- `last/` คือสถานะท้ายสุด ส่วน `best.json` ชี้ trained checkpoint ที่เลือกด้วย validation
  ทั้งสองอาจเป็นคนละน้ำหนัก และ `last/` ไม่ใช่ automatic resume
- Selected checkpoints แต่ละตัวมีโฟลเดอร์ของตนเอง; pointer ถูกเขียนหลัง weights/hash สำเร็จ
- Full data hashes ตรวจ inputs/history/labels; model manifest ตรวจ pinned assets;
  run contract ตรวจ config และ implementation ที่ใช้
- Stage 2 ต้องผูกกับ exact MF และ Stage 1 adapter ที่ใช้ตอนฝึก หากเปลี่ยน parent
  โปรแกรมต้องปฏิเสธการโหลด mapper เดิม แม้ dimensions จะยังตรงกัน
- โปรแกรมปฏิเสธ populated stage/result directories การทดลองใหม่ใช้ output_dir ใหม่
  ส่วน evaluation ซ้ำใช้ `--results-dir` ใหม่ได้โดยไม่ฝึกเพิ่ม
- `collm.run --check` ตรวจความพร้อมเท่านั้น การเอา `--check` ออกเริ่ม pipeline ฝึก

## เดินตามไฟล์จริง

1. เปิด `data/processed/train.jsonl` ตรวจ title, label และ history_timestamps
2. ดู `PromptDataset` ว่าข้อความถูก tokenize และตัด history อย่างไร
3. ดู `Collator` ว่า left padding เลื่อน user_pos/item_pos อย่างไร
4. ดู `MatrixFactorization.features()` และ `CoLLM.forward()` ว่า vectors ถูกแทรกตรงไหน
5. ดู `set_stage()` กับ trainable parameter checks ใน `train_stage()`
6. ตามจาก loss → backward → gradient clipping → optimizer update
7. ดูการสร้าง selection ใน `checkpoints.py` และไฟล์ `stage/best.json`
8. ดู parent verification ก่อน `evaluate.py` โหลด mapper และสร้าง predictions

ผลทดสอบล่าสุดอยู่ใน `results/verification.json` Tests ตรวจกลไกและความถูกต้องของ pipeline
ไม่ได้พิสูจน์ว่า CoLLM แนะนำหนังดีกว่า baselines ผล smoke ใช้แทนผลทดลองจริงไม่ได้

## การแบ่งงานและนำเสนอ

| ผู้รับผิดชอบหลัก | งานหลัก | สิ่งที่ต้องถ่ายทอดให้เพื่อน |
|---|---|---|
| คนที่ 1 | ข้อมูล, temporal split, history, MF | leakage และ collaborative vectors |
| คนที่ 2 | prompt, LoRA, CIE, gradients | tensor flow และการ freeze แต่ละ stage |
| คนที่ 3 | checkpoints, evaluation, ranking, report | provenance, selection และการตีความ AUC |

ทุกคนต้อง cross-review และตอบส่วนของเพื่อนได้ แบ่งความรับผิดชอบหลักได้ แต่ไม่แบ่งความเข้าใจ

โครงนำเสนอไม่เกิน 10 นาที: ปัญหา 1 นาที, โมเดล/สอง stages 2 นาที, ข้อมูล/leakage
1.5 นาที, สิ่งที่ปรับตามเครื่อง 1 นาที, ผลจริง 2 นาที, ranking/ข้อจำกัด/แบ่งงาน 2 นาที
ส่วนผลจริงยังต้องรอรอบทดลองที่ผู้ใช้สั่งให้เริ่มแล้วเท่านั้น

วันที่สอบตามเอกสารคือ 5 พฤศจิกายน 2026 ส่ง ZIP source code ก่อนเข้าเรียนตามช่องทาง
ที่อาจารย์กำหนด และระบุ backbone, subsets, input length, training budget,
constant learning rates, seed เดียว และ observed-rating evaluation ให้ชัดเจน
