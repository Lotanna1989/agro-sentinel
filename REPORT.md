# Technical Report — [AGRO-SENTINEL BY AGRORITHM]

**Team ID:** agro-sentinel 
**Domain:** Agriculture   
**Model:** Qwen2.5 3B Instruct-Q4_K_M

AgroSentinel - Offline farm Protection System and Livestock advisory

African Deep Tech Challenge 2026 - The Laptop LLM Challenge


Domain: Agriculture


NOTE ON NAMING - AgroSentinel (by AgroRithm) to align with the challenge Offline farm protection system and livestock advisory.


## 1. Problem Definition

Smallholder farmers across Nigeria's Middle Belt and agrarian states face three converging threats that existing digital tools do not address together:

1. **Farmer–herder conflict** — livestock straying or being driven into farmland triggers confrontations that regularly escalate into violence, crop destruction, and loss of life.
2. **Livestock early health advisory and market advisory** — An offline livestock health advisory to advise cattle herders of probable sicknesses and symptoms, first aid and vets that can assist. Also advisory on market and time to sell.
3. **Crop destruction and unauthorized intrusion** — theft and animal incursion go undetected until damage is already done, because there is no monitoring layer suited to rural, low-connectivity farms.

These three risks are usually treated as separate problems requiring separate tools. Agro-sentinel treats them as one problem: **a farm has no way to observe itself and warn its owner in time.** The underlying need — sense a risk, understand it, and alert someone before harm occurs — is identical whether the trigger is a person crossing a boundary, an animal approaching crops, or a livestock health or sale.

Cloud-based AI advisory tools do not fit this context. Rural Nigerian farms typically lack reliable electricity, consistent internet, and the ability to pay recurring API fees. Any system that depends on connectivity to function is a system that fails exactly when it's needed most — during a live intrusion or a rapidly rising pen temperature.

**Agro-sentinel is an on-device LLM system that observes farm conditions through computer vision and IoT sensors with RAG, reasons over that data using a locally grounded language model, and alerts the farmer — entirely offline. However we are using CHat-based infeence for now later we will expand to sensor based**

---

## 2. Constraints Identified

| Constraint | Detail | Design Implication |
|---|---|---|
| Memory ceiling | 7 GB RAM budget on ADTC Standard Laptop; exceeding it disqualifies the submission outright | Model must be quantized aggressively; RAG index must be small and efficient, not a large vector store |
| No discrete GPU | Intel UHD/Iris Xe or AMD Radeon integrated graphics only | All inference must run efficiently on CPU via llama.cpp |
| No cloud dependency | Model must run 100% offline with zero external network calls during evaluation | RAG retrieval, translation, and reasoning must all happen locally — no API calls anywhere in the pipeline |
| Toolchain lock | Only llama.cpp + GGUF weights are accepted for evaluation | Base model must be convertible to GGUF and run cleanly under llama.cpp |
| Real-world power/connectivity | Target deployment environment is rural Nigeria — intermittent power, no reliable cellular/internet | 
| Judging is model-only | Judges evaluate the .gguf file in isolation via LM Studio/Ollama, not the physical hardware build | Model's own reasoning quality, accuracy, and response format must carry the full weight of the accuracy score — hardware context supports the report/story but isn't directly benchmarked |

---

## 3. Design Decisions & Alternatives Considered

### 3.1 Base Model: Qwen2.5 3B Instruct (over Llama 3.2 3B)
Both models are GGUF/llama.cpp-compatible and fit within the memory budget at Q4_K_M quantization. Qwen2.5 3B was selected for stronger instruction-following and reasoning benchmarks at the same parameter class, which directly affects the accuracy component of scoring. Agric and nigerian based RAG was used alongside the qwen model

### 3.2 Cross-Disciplinary Integration: LLM + Offline RAG + Edge Sensing
Rather than a single-signal system, AgroRithm fuses three input types into one reasoning pipeline:
- **Computer vision and IoT sensors to be fully implemented later** (YOLOv8 Nano) for boundary and livestock/human detection

- **Offline RAG** over a local corpus of agricultural, market and veterinary advisory, and also regional conflict-pattern data

This satisfies the "load-bearing cross-disciplinary integration" requirement: the LLM's output is only as good as the RAG retrieval data feeding it, and the sensor data only becomes actionable once the LLM interprets and contextualizes it. Sensor will be fully implemented later. for now we are using the LLM and RAG

### 3.3 Why RAG Over Fine-Tuning
RAG and fine-tuning solve different problems: fine-tuning changes a model's underlying behavior permanently by baking knowledge into its weights, while RAG supplies verifiable, updatable facts at answer-time without altering the model itself. For AgroSentinel's core task — classifying risk against real numeric thresholds and routing alerts correctly — RAG is the better-fit tool, not merely a cheaper substitute: it keeps outputs grounded in citable, easily-updated source documents, avoids the overfitting risk that a small fine-tuning dataset would introduce (a particular concern given hidden test prompts are used to evaluate submissions), and keeps the base model file smaller and more efficient, supporting the efficiency score. Fine-tuning was evaluated and deliberately not used for Gate 1, with a light LoRA fine-tune on response formatting considered as a possible Phase 2 refinement rather than a requirement.

### 3.4 Event-Driven Inference for Thermal and Power Efficiency
An early automated-monitoring prototype ran LLM inference on a fixed short interval (every 10 seconds) regardless of whether any reading was actually risky. Testing this on target-equivalent hardware surfaced a real thermal and power cost: sustained CPU inference at that frequency produced noticeable heat and fan load, directly relevant to the S_total scoring formula's thermal penalty term. The sesnors will be built much later for now we wull use RAG and LLM.

The architecture was revised to separate two concerns that had been incorrectly tied to one timer: sensor polling (cheap, checking a value) and LLM inference (expensive, the actual compute cost). The current design polls simulated sensor streams frequently (every 30 seconds) but only invokes the LLM when a reading crosses a real risk threshold drawn from the corpus, and applies a cooldown period (15 minutes) per location to avoid re-running inference repeatedly on an unchanging condition. This is both more efficient on constrained hardware and a more realistic production behavior: a real deployment does not need a fresh AI assessment every few seconds on a pen that is already known to be at risk.


---

## 4. System Flow

**Chat Inference** → The user prompts the model.

**Understand** → offline RAG retrieves relevant grounding (veterinary thresholds,conflict, market and health advicory, regional patterns); the on-device LLM reasons over combined context.

**Predict** → The model outputs a risk classification with a short natural-language explanation (e.g., livestock health triage or market value and time to sell or a boundary intrusion pattern consistent with past incidents, the risks and what to do as a farmer).



**Learn** → Local event logs refine the RAG knowledge base over time. No data leaves the device.

---

## 5. Tools & Benchmarks

- **Base model**: Qwen2.5 3B Instruct
- **Quantization**: Q4_K_M via llama.cpp
- **Format**: GGUF
- **Retrieval(RAG)**: local vector index over agricultural/veterinary/conflict-pattern corpus


- **Target hardware**: ADTC Standard Laptop (Intel i5 10th–12th gen, 8 GB RAM, 1TB HDD, windows, integrated graphics, low budget laptop)

- **VERY IMPORTANT SETUP FOR AGRO-SENTINEL**: After running the Qwen base model, NEXT you run python setup_offline_model.py once (requires internet) to cache the embedding model locally. All subsequent runs of build_index.py and rag_pipeline.py operate fully offline, then after you run rag_pipeline you can now chat infer the model for agro based responses.

**Benchmark results (measured via official ADTC profiler, participant mode, full run including accuracy):**

| Metric | Result |
|---|---|
| Throughput (tokens/sec, generation) | 8.81 |
| First token latency | 11,699.92 ms |
| Peak RAM (peak_rss) | 3,308.43 MB (~3.23 GB, well under the 7 GB budget) |
| Steady-state RAM | 3,185.61 MB |
| CPU thermal (p99) | 93.5% CPU utilization, **throttled: false** |
| Accuracy (ARC-Easy, 50 samples) | 0.78 (arc_norm) |
| params_match | true (3.397B params confirmed, Qwen2 architecture) |

No thermal throttling was observed at any point during profiling, and peak memory usage left roughly 3.7 GB of headroom under the 7 GB disqualification ceiling.

---

## 6. Bonus Claims


- **Budget Laptop Compatibility**: Quantized model designed to run within 7 GB RAM ceiling on $150–$500 hardware (targets Budget Profile bonus, +10%)
- **African Use Case Depth**: Grounded in real Nigerian farmer-herder conflict data and rural livestock farm, health and market conditions, not a generic agricultural chatbot

---

## 7. Locked Test Prompts (Gate 1 Submission)

**Test Prompt 1 (Livestock/Conflict pillar, farmer/herder voice):**
"I am a farmer in Makurdi and cattle herders intruded, what immediate steps can I take to prevent a repeat and who can I contact if the situation becomes unsafe?"

**Test Prompt 2 (Livestock Health + Market Advisory pillar, farmer voice):**
"I have a ram in Kano weighing about 40kg, and Sallah is coming soon. It has a fever and some sores around its mouth. Should I still sell it for Sallah, and what would it likely be worth?"

---

### Deterministic Calculation Principle (Health + Market Advisory Architecture)

Following the design principle that a small local LLM should not be responsible for arithmetic or act as the source of market data, the health+market advisory flow separates calculation from reasoning:

```
        Animal weight (380kg) + health status (reduced appetite)
                              |
                              v
        App layer: deterministic price estimate
        380kg x ~N850/kg baseline = ~N323,000 (pre-adjustment)
                              |
                              v
              RAG retrieval (2 corpus documents):
        cattle_market_advisory.txt  +  livestock_heat_stress.txt
                              |
                              v
                    Local LLM (Qwen2.5 3B)
                              |
                              v
        Synthesizes: price estimate + health signal + season
        -> "Resolve health concern before selling" recommendation
```

This mirrors the architecture used for the Prompt 1 (conflict/geofence) pillar: structured/calculated data feeds into RAG-grounded LLM reasoning, rather than asking the model to invent numbers or act as a database. The model's role stays consistent across both pillars: retrieve -> understand -> synthesize -> explain -> advise.

---

## 8. Demo Scenarios

1. **Crop/intrusion detection**: An animal or unauthorized presence approaches a monitored crop field boundary → system detects, classifies risk, generates alert.
2. **Livestock Market advisory and health advisory**: 

Two scenarios were chosen deliberately to demonstrate that one underlying system — one model, one reasoning pipeline — protects both crops/security and livestock/environmental wellbeing, without requiring three separate purpose-built tools. 

## 9. HOW TO SETUP, STARTUP AND RUN THE AGRO-SENTINEL OFFLINE EXECUTION FLOWCHART**: 
1. First you will download qwen by running download_model.sh and also download the embedding model by running setup_offline_model.py so it can cache locally. 
2. Now OFFLINE, Next go to powershell and point to the local qwen file location on your laptop and then run qwen using this command .llama-server.exe -hf Qwen/Qwen2.5B-3B-Instruct-GGUF:Q4_K_M --port 8080.
3. Then from here you can run build_index.py and finally rag_pipeline.py all OFFLINE and LOCALLY. All inference is LOCAL.


```
             START
                │
                ▼
      🌐 Internet Connection
                │
        ┌───────┴────────┐
        ▼                ▼
 Download Qwen     Run setup_offline_model.py
        │                │
        ▼                ▼
 Store & run Qwen  Cache Embedding Model
        │                │
        └───────┬────────┘
                ▼
        🔌 Internet No Longer Needed
                │
                ▼
        Run build_index.py
                │
                ▼
      Build Local ChromaDB
       Knowledge Index
                │
                ▼
       Run rag_pipeline.py
                │
                ▼
          👤 User Query
                │
                ▼
       Cached Embedding Model
                │
                ▼
       Search Local ChromaDB
                │
                ▼
      Retrieve Relevant Context
                │
                ▼
        Local Qwen2.5 Model
         (llama-server)
                │
                ▼
        🤖 Generate Response
                │
                ▼
          👤 User Response
                │
                ▼
          100% OFFLINE

```
