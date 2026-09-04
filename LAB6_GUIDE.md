# Lab 6: Occupancy Grid Mapping (OGM) 4x4 Grid — RoboMaster EP

## 1. วัตถุประสงค์ (Objectives)
1. เพื่อเข้าใจและประยุกต์ใช้หลักการ **Occupancy Grid Mapping (OGM)**
2. ควบคุมหุ่นยนต์ DJI RoboMaster ให้สร้างแผนที่ตารางขนาด **$4 \times 4$ กริด** (ขนาดช่องละ $60 \times 60\text{ cm}$) แบบเรียลไทม์
3. ล็อกมุม Gimbal ให้อยู่ในแนวขนานพื้น (Pitch=0°, Yaw=0°) เพื่อให้เซนเซอร์ ToF ด้านหน้าอ่านค่าได้ถูกต้องโดยไม่ต้องขยับ Gimbal
4. ใช้เซนเซอร์ ToF (หน้า) ร่วมกับ Digital IR Obstacle Sensors แบบ I/O (ซ้าย/ขวา ผ่าน `sensor_adaptor.get_io()`) อัปเดตความน่าจะเป็นของการมีสิ่งกีดขวางด้วย **Bayes' Rule / Log-Odds Update**
5. บันทึกผลลัพธ์ลงในไฟล์ CSV, TXT และภาพแผนที่ความร้อน (PNG)

---

## 2. ทฤษฎีและหลักการ (Theory: Log-Odds Occupancy Grid Mapping)

### 2.1 นิยามของตารางกริด (Grid Definition)
- พื้นที่แบ่งเป็นกริดขนาด $4 \times 4$ ช่อง พิกัด $(x, y)$ โดยที่ $x, y \in \{0, 1, 2, 3\}$
- ขนาดแต่ละช่อง: $60\text{ cm} \times 60\text{ cm}$ (พื้นที่รวม $2.4\text{ m} \times 2.4\text{ m}$)
- แต่ละเซลล์ $m_i$ มีค่าสถานะความน่าจะเป็นของการมีสิ่งกีดขวาง $P(m_i) \in [0, 1]$:
  - $P(m_i) = 0.5$ : **Unknown** (ยังไม่เคยตรวจพบ / ข้อมูลเริ่มต้น Prior)
  - $P(m_i) < 0.5$ : **Free** (พื้นที่ว่าง ไม่มีสิ่งกีดขวาง)
  - $P(m_i) > 0.5$ : **Occupied** (มีสิ่งกีดขวาง / กำแพง)

### 2.2 สูตร Log-Odds และการอัปเดตแบบเบย์ (Bayesian Update)
เพื่อหลีกเลี่ยงปัญหาค่าทศนิยมตกขอบ (numerical underflow/overflow) จะใช้ตัวแทนแบบ **Log-Odds**:
$$l(m_i) = \ln\left(\frac{P(m_i)}{1 - P(m_i)}\right)$$

- ค่าเริ่มต้น (Prior): เมื่อ $P_0 = 0.5 \implies l_0 = \ln(1) = 0.0$
- การอัปเดตเมื่อมีข้อมูลเซนเซอร์ใหม่ $z_t$:
  $$l_t(m_i) = l_{t-1}(m_i) + \text{inv\_sensor}(z_t, m_i) - l_0$$
  เนื่องจาก $l_0 = 0$ จะได้:
  $$l_t(m_i) = l_{t-1}(m_i) + l_{\text{sensor}}$$

- **Inverse Sensor Model**:
  - เมื่อเซนเซอร์ตรวจว่าพื้นที่ว่าง (Free space):
    $$P_{\text{free}} = 0.35 \implies l_{\text{free}} = \ln\left(\frac{0.35}{0.65}\right) \approx -0.619$$
  - เมื่อเซนเซอร์ตรวจพบสิ่งกีดขวาง (Occupied cell):
    $$P_{\text{occ}} = 0.85 \implies l_{\text{occ}} = \ln\left(\frac{0.85}{0.15}\right) \approx +1.735$$

- การแปลงค่า Log-Odds กลับมาเป็นความน่าจะเป็น (Probability):
  $$P(m_i) = \frac{1}{1 + \exp(-l(m_i))}$$

---

## 3. สถาปัตยกรรมและไฟล์ในโปรเจกต์ (Project Structure)

```
c:\robotproject\robot_sahapong\lab6\
├── LAB6_GUIDE.md           # คู่มือและทฤษฎี (เอกสารนี้)
├── requirements.txt        # ไลบรารีที่จำเป็นสำหรับติดตั้งใน Python
├── config.py               # ค่าพารามิเตอร์ระบบ (Grid 4x4, พอร์ต Digital IR IO, Exit Cell)
├── ogm.py                  # คลาส OccupancyGridMap จัดการ Bayes Log-Odds และ Plot แผนที่
├── sensors.py              # เซนเซอร์: ควบคุม ToF + Gimbal Lock (นิ่ง) + Digital IR I/O
├── explorer.py             # ระบบสำรวจเขาวงกตที่ไม่รู้แผนที่มาก่อนเพื่อหาทางออกอัตโนมัติ
├── dashboard.py            # หน้าจอ Real-Time GUI Dashboard (OpenCV)
├── test_digital_ir.py      # สคริปต์ทดสอบเซนเซอร์ Digital IR IO บนหุ่นจริง
└── lab6_main.py            # สคริปต์หลัก ควบคุม Step-by-Step, Autonomous Exploration & Dashboard
```

---

## 4. วิธีการใช้งานและการรัน (How to Run)

### 4.1 ให้หุ่นยนต์สำรวจหาทางออกเองอัตโนมัติ (Autonomous Exploration & Real-Time Dashboard)
หุ่นยนต์จะเริ่มที่จุด $(0, 0)$ และสำรวจพื้นที่ที่ไม่รู้แผนที่มาก่อนด้วยตนเอง จนกระทั่งหาทางออกที่ $(3, 3)$ พบ พร้อมแสดงผล Dashboard แบบ Real-time:

- **รันบนหุ่นยนต์จริง (Real Robot):**
  ```powershell
  .\.venv\Scripts\python.exe robot_sahapong\lab6\lab6_main.py --real --explore
  ```

- **รันในโหมดจำลอง (Mock Simulation):**
  ```powershell
  .\.venv\Scripts\python.exe robot_sahapong\lab6\lab6_main.py --mock --explore
  ```

### 4.2 รันแบบควบคุมทีละก้าว (Interactive Control Mode):
```powershell
.\.venv\Scripts\python.exe robot_sahapong\lab6\lab6_main.py --real
```
ปุ่มควบคุมในโหมด Interactive:
- `w` : เดินหน้า 1 ช่อง ($60\text{ cm}$)
- `a` : เลี้ยวซ้าย 90°
- `d` : เลี้ยวขวา 90°
- `s` : สแกนและอัปเดต OGM ณ ตำแหน่งปัจจุบัน
- `e` : สั่งให้หุ่นสำรวจหาทางออกเองอัตโนมัติทันที
- `p` : เซฟรูปแผนที่และข้อมูลปัจจุบันทันที
- `q` : จบการทดลองและบันทึกผลลัพธ์ทั้งหมด
