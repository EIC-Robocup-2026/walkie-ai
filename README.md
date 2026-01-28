# Walkie-AI

โปรเจกต์นี้คือระบบ AI Agent ที่มีความสามารถในการรับรู้หลายรูปแบบ (Multimodal) ทั้งเสียง (Voice), ภาพ (Vision) และการโต้ตอบ (Interaction) โดยมีการออกแบบโครงสร้างแบบ **Modular Architecture** เพื่อให้ง่ายต่อการขยายความสามารถ (Scalability) และการดูแลรักษา (Maintainability)

## 📂 Project Structure Overview

โครงสร้างของโปรเจกต์ถูกแบ่งตามหน้าที่การทำงาน (Separation of Concerns) ดังนี้:

```text
project_root/
├── config/             # การตั้งค่าระบบ (Configuration)
├── data/               # ข้อมูล Local (Database files, temp images)
├── src/                # Source code หลักของระบบ
│   ├── agent/          # [The Brain] Logic หลักในการตัดสินใจ
│   ├── database/       # [Memory] การจัดการข้อมูลและ Persistence
│   ├── interface/      # [I/O] การรับ/ส่งข้อมูลเสียง (ASR/TTS)
│   ├── tools/          # [Skills] ความสามารถพิเศษ (RAG, Navigation)
│   ├── vision/         # [Eyes] การประมวลผลภาพ (Image Model)
│   └── utils/          # ฟังก์ชันช่วยเหลือทั่วไป
├── tests/              # Unit tests
├── main.py             # Entry point ของโปรแกรม
└── requirements.txt    # Python dependencies
````

-----

## 🏗️ Module Descriptions

รายละเอียดของแต่ละ Module ภายใน `src/` และความสัมพันธ์กับ System Diagram:

### 1\. `src/agent/` (The Brain)

เปรียบเสมือน "สมอง" ของระบบ ทำหน้าที่เป็น Orchestrator คอยรับ Input และตัดสินใจว่าจะทำอะไรต่อไป

  * **Core Role:** รับข้อความจาก ASR, ตัดสินใจเรียกใช้ Tools, และส่งคำตอบกลับไปที่ TTS
  * **Files:**
      * `core.py`: Main agent with LangChain ReAct framework
      * `memory.py`: Conversation history and context management
      * `state.py`: Agent status tracking (idle, processing, speaking)
      * `mcp_server.py`: MCP server exposing agent as tools

### 2\. `src/interface/` (Input/Output)

ส่วนติดต่อกับโลกภายนอก (Human-Computer Interaction)

  * **ASR (Automatic Speech Recognition):** แปลงเสียงพูดเป็นข้อความ
  * **TTS (Text-to-Speech):** แปลงข้อความตอบกลับเป็นเสียงพูด
  * **Files:** `asr/`, `tts/`

### 3\. `src/tools/` (Capabilities)

เครื่องมือต่างๆ ที่ Agent สามารถเรียกใช้งานได้ (Function Calling)

  * **`rag/` (Retrieval-Augmented Generation):** ค้นหาข้อมูลจากความรู้ที่มีอยู่ เพื่อตอบคำถามให้แม่นยำขึ้น
  * **`navigation/`:** ควบคุมการเคลื่อนที่ หรือคำนวณเส้นทาง (Planner/Controller)
  * **`action_policy/`:** กฎเกณฑ์ความปลอดภัย หรือ Logic ในการตัดสินใจเฉพาะทาง

### 4\. `src/vision/` (The Eyes)

ส่วนประมวลผลข้อมูลภาพ แยกเป็นอิสระเพื่อประสิทธิภาพ

  * **Image Model:** ใช้ Model เพื่อแปลงภาพเป็น Vector (Embeddings) หรือวิเคราะห์สิ่งที่เห็น
  * **Flow:** Image Capture -\> Image Model -\> Database

### 5\. `src/database/` (Memory & Storage)

ศูนย์กลางการจัดเก็บข้อมูลทั้งหมด ตามกล่องขวาสุดของ Diagram

  * **Schema:** กำหนดโครงสร้างข้อมูลให้ชัดเจน (Pydantic Models) เช่น:
      * `Coordinate` / `Pose`
      * `Semantic Location` (e.g., "Kitchen")
      * `Image Embeddings` (Vector search)
  * **Connection:** จัดการการเชื่อมต่อกับ SQL หรือ Vector Database

-----

## 🚀 Getting Started

โปรเจกต์นี้ใช้ **Conda** environment สำหรับการจัดการ dependencies

### Prerequisites

  * Python 3.11+
  * Conda (Miniconda or Anaconda)
  * API Keys: OpenAI, ElevenLabs, Google Cloud

### Installation

1.  Clone repository:

    ```bash
    git clone <your-repo-url>
    cd walkie-ai
    ```

2.  Create Conda environment:

    ```bash
    conda create -n eic python=3.11 -y
    conda activate eic
    ```

3.  Install dependencies:

    ```bash
    pip install -e .
    ```

4.  Environment Setup:
    Copy `.env.example` to `.env` and add your API keys:

    ```bash
    cp .env.example .env
    # Edit .env with your keys
    ```

### Usage

**Interactive Mode:**
```bash
python main.py --mode interactive
```

**Text Command:**
```bash
python main.py --mode text --command "Your command here"
```

**Voice Mode:**
```bash
python main.py --mode voice --audio input.wav --output response.mp3
```

**MCP Server (Recommended for integration):**
```bash
python -m src.agent.mcp_server
# Or test it:
python test_mcp_server.py
```

-----

## 🛠️ Tech Stack

  * **Language:** Python 3.11
  * **Agent Framework:** LangChain (ReAct agent)
  * **LLM:** OpenAI GPT-4
  * **TTS:** ElevenLabs
  * **STT:** Google Cloud Speech-to-Text
  * **Audio Processing:** Silero VAD, sounddevice
  * **Integration:** MCP (Model Context Protocol)
