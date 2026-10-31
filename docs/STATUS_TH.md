# สถานะก่อนเริ่มเทรนจริง

## จุดเริ่มที่เตรียมไว้

ดับเบิลคลิก **START_TRAIN.cmd** รอการตรวจความพร้อม แล้วกด **Train** เป็นขั้นตอนสุดท้าย
หน้าต่างจะฝึกทั้งสามโมเดล ประเมิน และสร้างรายงาน HTML/CSV/กราฟ/ZIP ให้อัตโนมัติ
คู่มือใช้งาน: [TRAIN_GUIDE_TH.md](TRAIN_GUIDE_TH.md)
โค้ดต้อง commit และ push แล้วก่อนเริ่มรอบจริง ตามคำขอของผู้ใช้

**ยังไม่ได้เริ่มรอบจริง** ผู้ใช้กำชับให้เตรียมโค้ดและตรวจแบบสั้นก่อน
อนุญาตเฉพาะ preparation, preflight, unit tests, short benchmarks และ smoke checks

## สิ่งที่จัดเตรียมและปรับแล้ว

- Environment Python 3.11 แยกใน `.venv` พร้อม PyTorch สำหรับ Intel XPU
- MovieLens-1M และ Qwen2-0.5B ที่ระบุ revision แน่นอน พร้อมตรวจ SHA256
- Temporal splits 33,891 / 10,401 / 7,331 interactions และประวัติที่เกิดก่อนเป้าหมาย
- MF, text-only LoRA, CIE mapper, AUC, ranking และตัวสร้างรายงาน
- ลบ history IDs ที่ไม่ได้ใช้, attributes/imports ที่ไม่ได้ใช้ และการคำนวณ No logit ที่ทิ้ง
- แยก labels ออกจาก model inputs และ tokenize ข้อความคงที่เพียงครั้งเดียว
- ตรวจ config, full data content และ model assets ก่อนโหลด checkpoints
- Selected checkpoints แยกโฟลเดอร์ ไม่เขียนทับ; `best.json` ชี้ตัวที่เลือกหลังบันทึกสำเร็จ
- ตรวจว่า mapper ใช้ MF และ LoRA parents ชุดเดียวกับตอนฝึก
- ปฏิเสธ stage/results เดิมเมื่อรันซ้ำ ลดโอกาสปะปนไฟล์จากคนละการทดลอง
- รองรับ last snapshots และสถานะเมื่อหยุด/หมดเวลา ตัวเปิดเทรนใช้ขั้นที่เสร็จแล้วต่อได้
  ส่วนขั้นที่ค้างเริ่มใหม่ ไม่ใช่ optimizer-step resume
- สร้างชื่อรอบและสำเนา config อัตโนมัติ มี log และป้องกันหลายงานใช้เครื่องพร้อมกัน
- มีปุ่มหยุดอย่างปลอดภัย และรายงานสถานะ time_limit แยกจากการทำครบแผน
- คู่มืออ่านโค้ดและเครื่องมือจัด source ZIP

ผลตรวจล่าสุดอ้างอิง [results/verification.json](../results/verification.json)
ผลในโฟลเดอร์ smoke เป็นการตรวจ pipeline ไม่ใช่หลักฐานเปรียบเทียบประสิทธิภาพโมเดล

## ตรวจได้ทันทีโดยไม่เริ่มเทรน

```powershell
.\.venv\Scripts\python.exe -m collm.run --check --config configs/local.yaml
```

คำสั่งนี้ตรวจ pinned assets, selected records, prompts, device และ output path
ไม่โหลดน้ำหนัก LLM เพื่อฝึก ไม่สร้าง checkpoint และไม่ทำ optimizer updates

ถ้าจะรัน smoke ซ้ำ ให้คัดลอก `configs/smoke.yaml` และเลือก `output_dir` ใหม่
โปรแกรมตั้งใจปฏิเสธโฟลเดอร์เดิม ไม่ต้องลบผลเก่าเพื่อให้รันต่อได้

## ยังไม่ได้ทำ

- ยังไม่มีการฝึกจริงตาม `configs/local.yaml` และยังไม่มีผลจริงสำหรับส่งงาน
- ยังไม่ขยายเป็น Qwen2-1.5B, full-size experiment หรือหลาย seeds
- ยังไม่สร้างสไลด์ที่อ้างผลทดลองจริง และยังไม่ส่งไฟล์ขึ้น MyCourseVille

## เหลือขั้นตอนสุดท้าย

ผู้ใช้ตรวจค่าที่แสดงในหน้าต่าง แล้วกด **Train** เมื่อต้องการเริ่มจริง
ขนาดเริ่มต้นถูกเตรียมไว้ที่ train/validation/test 4,096 / 512 / 1,024 ตามแผนเดิม
ไม่ต้องแก้คำสั่ง สร้างชื่อรอบ หรือเรียกแต่ละ stage เอง

ตกลงแล้ว: ใช้เครื่องนี้, Qwen2-0.5B เป็นตัวแรก, MF/TALLRec/CoLLM-MF,
MovieLens-1M, AUC และเพดาน 6 ชั่วโมงต่อคำสั่ง ไม่ต้องยืนยันเรื่องเหล่านี้ซ้ำ

## เมื่อได้รับคำสั่งให้เริ่ม

- ตัวเปิดเทรนสร้าง `output_dir` ใหม่ให้การทดลองแต่ละรอบ และตรวจ preflight/Git ก่อน
- Pipeline เรียก MF → Stage 1 → Stage 2 → evaluation ตามลำดับ โดยแต่ละ stage มี timer ของตนเอง
- เลือก checkpoints ด้วย validation เท่านั้น; รายงานเวลารวมและจำนวน steps ที่ทำได้จริง
- ผล evaluation เดิมไม่ถูกเขียนทับ หากประเมินซ้ำให้ใช้ `--results-dir results/<ชื่อใหม่>`
- ถ้าหยุดระหว่างทาง ให้ตรวจ `last/`, `summary.json`, `status.json` และ selected `best.json`
  ตัวเปิดเทรนนำขั้นที่เสร็จและเข้ากันได้กลับมาใช้ในโฟลเดอร์ใหม่ ส่วนขั้นที่ค้างเริ่มใหม่

Benchmark ที่เคยวัดได้ประมาณครึ่งวินาทีต่อ XPU microbatch ขนาด 4 เป็นการวัดสั้นในอดีต
ไม่ใช่เวลาที่รับประกันหลัง cleanup หรือเวลารวมของรอบจริง และไม่ใช่แผนให้รันครบ 6 ชั่วโมง
