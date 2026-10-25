/**
 * @file BOARD_A.ino
 * @brief Unified ESP32-S3 Federated Learning Client — Device Under Test (DUT)
 *
 * Hardware:
 *   - Board A: ESP32-S3 Dev Board (Powered through INA238 Shunt on 5V rail)
 *
 * Pinout Connections (Cross-wired to Board B Logger):
 *   - Board A GPIO 10 (Output) --> Board B GPIO 6  (TRAIN Strobe)
 *   - Board A GPIO 11 (Output) --> Board B GPIO 7  (COMM Strobe)
 *   - Board A GPIO 12 (Output) --> Board B GPIO 8  (Sub-phase Bit 0)
 *   - Board A GPIO 13 (Output) --> Board B GPIO 9  (Sub-phase Bit 1)
 *   - Board A GPIO 14 (Output) --> Board B GPIO 18 (Sub-phase Bit 2)
 *   - Board A GND              --> Common GND (Board B GND & INA238 GND)
 *
 * CRITICAL HARDWARE REQUIREMENT:
 *   - 100 kOhm external pull-down resistors MUST be installed on all 5 marker lines
 *     (GPIO 10, 11, 12, 13, 14) to GND (replaces legacy 10 kOhm spec).
 *   - Load Bias Analysis:
 *     At 10 kOhm, asserting 4 lines simultaneously (e.g., Code 111 + TRAIN) at 3.3V draws:
 *       4 * (3.3V / 10 kOhm) = 1.32 mA (~4.36 mW at 3.3V, ~6.6 mW on 5V rail).
 *     This introduces a measurable artificial bias during training.
 *     At 100 kOhm, this current drops by 10x to 0.132 mA (~0.44 mW), well below
 *     the INA238 measurement noise floor (~0.5 mA RMS).
 *
 * Sub-Phase State Encoding (3-Bit Hardware Code):
 *   0 (000): IDLE / BASELINE (DUT quiescent, no train or comm active)
 *   1 (001): DNS / ARP LOOKUP
 *   2 (010): TCP SYN / CONNECT
 *   3 (011): HTTP REQUEST / HEADERS
 *   4 (100): HTTP PAYLOAD TRANSFER (3,120 B model weights)
 *   5 (101): SERVER RESPONSE WAIT / ACK
 *   6 (110): LOCAL SGD TRAINING (TRAIN active)
 *   7 (111): PRE-RUN CLOCK SYNC STROBE (SYNC_CALIBRATION, run-start calibration pulse,
 *            fired ONCE in setup(), excluded from energy integration)
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <HTTPClient.h>
#include <esp_sleep.h>
#include <esp_wifi.h>

static WiFiUDP udp_sync;
#define SYNC_UDP_PORT 9999

/* ===================================================================
 * Wi-Fi & Server Credentials
 * =================================================================== */
const char* WIFI_SSID      = "YOUR_SSID";        // set before flashing
const char* WIFI_PASS      = "YOUR_PASSWORD";    // set before flashing
const char* SERVER_IP      = "192.168.1.100";    // IP of the host running host/server/aggregate.py
const uint16_t SERVER_PORT = 8080;

/* ===================================================================
 * GPIO Phase & 3-Bit Sub-Phase Hardware Marker Pins
 * =================================================================== */
#define PIN_GPIO_TRAIN     10   // Backward-compatible TRAIN strobe
#define PIN_GPIO_COMM      11   // Backward-compatible COMM strobe
#define PIN_PHASE_BIT0     12   // Sub-phase Bit 0
#define PIN_PHASE_BIT1     13   // Sub-phase Bit 1
#define PIN_PHASE_BIT2     14   // Sub-phase Bit 2

inline void set_phase_code(uint8_t code) {
    digitalWrite(PIN_PHASE_BIT0, (code & 1) ? HIGH : LOW);
    digitalWrite(PIN_PHASE_BIT1, (code & 2) ? HIGH : LOW);
    digitalWrite(PIN_PHASE_BIT2, (code & 4) ? HIGH : LOW);
    digitalWrite(PIN_GPIO_TRAIN, (code == 6) ? HIGH : LOW);
    digitalWrite(PIN_GPIO_COMM,  (code >= 1 && code <= 5) ? HIGH : LOW);
}

/* ===================================================================
 * Measurement & Protocol Configurations
 * =================================================================== */
// Mode A (false): Standard back-to-back execution (original benchmark pipeline)
// Mode B (true): Clean Tail Isolation Mode (inserts 60 ms >= 50 ms settling delay between all phases)
#define CLEAN_TAIL_MEASUREMENT_MODE true
#define INTER_PHASE_SETTLE_MS       60
static int g_inter_phase_settle_ms = INTER_PHASE_SETTLE_MS;

// Sleep mode selection:
// 0 = Active Standby (No sleep, WiFi.setSleep(false))
// 1 = Modem Sleep (WiFi.setSleep(true) during idle)
// 2 = Light Sleep (802.11 Power Save, CPU clock-gated, association retained)
// 3 = Deep Sleep (RTC only, cold reboot + Wi-Fi re-association)
enum SleepMode {
    SLEEP_ACTIVE_STANDBY = 0,
    SLEEP_MODEM_SLEEP    = 1,
    SLEEP_LIGHT_SLEEP    = 2,
    SLEEP_DEEP_SLEEP     = 3
};

#define ACTIVE_SLEEP_MODE   SLEEP_ACTIVE_STANDBY
#define ROUND_PERIOD_MS     10000UL // 10 s configurable round period

// Hardware calibration mode: enables 6-second marker pull-down load bias calibration run in setup()
#define ENABLE_PULLDOWN_BIAS_CALIBRATION false

/* ===================================================================
 * Model Dimensions & FL Hyperparameters
 * =================================================================== */
#define FL_NUM_INPUTS       64                                      // Backbone output features
#define FL_NUM_CLASSES      12                                      // KWS classes
#define FL_NUM_WEIGHTS      (FL_NUM_INPUTS * FL_NUM_CLASSES)        // 768 float params
#define FL_NUM_BIASES       FL_NUM_CLASSES                          // 12 float params
#define FL_NUM_PARAMS       (FL_NUM_WEIGHTS + FL_NUM_BIASES)        // 780 params total
#define FL_PARAM_BYTES      (FL_NUM_PARAMS * sizeof(float))         // 3,120 bytes

#define FL_BATCH_SIZE       16                                      // Exact mini-batch size
#define FL_LEARNING_RATE    0.01f                                   // Exact SGD learning rate
#define FL_NUM_SAMPLES      100                                     // Samples per virtual client

/* ===================================================================
 * Global Buffers & Model State
 * =================================================================== */
static float g_weights[FL_NUM_WEIGHTS];
static float g_biases[FL_NUM_BIASES];
static float g_grad_w[FL_NUM_WEIGHTS];
static float g_grad_b[FL_NUM_BIASES];

static float   g_features[FL_BATCH_SIZE * FL_NUM_INPUTS];
static float   g_logits[FL_BATCH_SIZE * FL_NUM_CLASSES];
static float   g_probs[FL_BATCH_SIZE * FL_NUM_CLASSES];
static uint8_t g_labels[FL_BATCH_SIZE];

static uint8_t g_model_buf[FL_PARAM_BYTES];

static uint32_t g_rng_state = 42;
static inline uint32_t xorshift32(void) {
    uint32_t x = g_rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng_state = x;
    return x;
}

static inline float rand_uniform(void) {
    return (float)(xorshift32() & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

/* ===================================================================
 * Model Training Math (Forward, Backward, SGD)
 * =================================================================== */
void generate_batch(int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < FL_NUM_INPUTS; j++) {
            g_features[i * FL_NUM_INPUTS + j] = (rand_uniform() - 0.5f) * 2.0f;
        }
        g_labels[i] = (uint8_t)(xorshift32() % FL_NUM_CLASSES);
    }
}

void forward_pass(int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        const float *xi = &g_features[i * FL_NUM_INPUTS];
        float *li = &g_logits[i * FL_NUM_CLASSES];

        for (int c = 0; c < FL_NUM_CLASSES; c++) {
            float sum = g_biases[c];
            for (int j = 0; j < FL_NUM_INPUTS; j++) {
                sum += xi[j] * g_weights[j * FL_NUM_CLASSES + c];
            }
            li[c] = sum;
        }

        float max_val = li[0];
        for (int c = 1; c < FL_NUM_CLASSES; c++) {
            if (li[c] > max_val) max_val = li[c];
        }
        float sum_exp = 0.0f;
        float *pi = &g_probs[i * FL_NUM_CLASSES];
        for (int c = 0; c < FL_NUM_CLASSES; c++) {
            pi[c] = expf(li[c] - max_val);
            sum_exp += pi[c];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int c = 0; c < FL_NUM_CLASSES; c++) {
            pi[c] *= inv_sum;
        }
    }
}

float compute_loss(int batch_size) {
    float loss = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        float p = g_probs[i * FL_NUM_CLASSES + g_labels[i]];
        if (p < 1e-7f) p = 1e-7f;
        loss -= logf(p);
    }
    return loss / (float)batch_size;
}

void backward_pass(int batch_size) {
    memset(g_grad_w, 0, sizeof(g_grad_w));
    memset(g_grad_b, 0, sizeof(g_grad_b));
    const float inv_b = 1.0f / (float)batch_size;

    for (int i = 0; i < batch_size; i++) {
        const float *xi = &g_features[i * FL_NUM_INPUTS];
        const float *pi = &g_probs[i * FL_NUM_CLASSES];
        const uint8_t y = g_labels[i];

        for (int c = 0; c < FL_NUM_CLASSES; c++) {
            float dl = (pi[c] - (c == y ? 1.0f : 0.0f)) * inv_b;
            g_grad_b[c] += dl;
            for (int j = 0; j < FL_NUM_INPUTS; j++) {
                g_grad_w[j * FL_NUM_CLASSES + c] += xi[j] * dl;
            }
        }
    }
}

void sgd_step(float lr) {
    for (int i = 0; i < FL_NUM_WEIGHTS; i++) g_weights[i] -= lr * g_grad_w[i];
    for (int i = 0; i < FL_NUM_BIASES; i++)  g_biases[i]  -= lr * g_grad_b[i];
}

void train_local_model(int epochs, int client_id) {
    const int num_batches = (FL_NUM_SAMPLES + FL_BATCH_SIZE - 1) / FL_BATCH_SIZE;

    set_phase_code(6); // Phase 6: LOCAL SGD TRAINING
    for (int e = 0; e < epochs; e++) {
        g_rng_state = 42 + (uint32_t)client_id;
        float epoch_loss = 0.0f;

        for (int b = 0; b < num_batches; b++) {
            int bs = FL_BATCH_SIZE;
            if (b == num_batches - 1 && (FL_NUM_SAMPLES % FL_BATCH_SIZE) != 0) {
                bs = FL_NUM_SAMPLES % FL_BATCH_SIZE;
            }
            generate_batch(bs);
            forward_pass(bs);
            epoch_loss += compute_loss(bs) * bs;
            backward_pass(bs);
            sgd_step(FL_LEARNING_RATE);
        }
        epoch_loss /= (float)FL_NUM_SAMPLES;
    }
    set_phase_code(0); // Return to IDLE
}

/* ===================================================================
 * HTTP Communications with FL Server (Sub-phase Instrumented)
 * =================================================================== */
bool fetch_server_config(int &round, int &epochs, int &client_id, bool &is_finished) {
    set_phase_code(2); // Phase 2: TCP CONNECT
    WiFiClient client;
    HTTPClient http;
    String url = String("http://") + SERVER_IP + ":" + SERVER_PORT + "/config";

    if (!http.begin(client, url)) {
        set_phase_code(0);
        return false;
    }

    set_phase_code(3); // Phase 3: HTTP REQUEST / HEADERS
    int code = http.GET();
    if (code != 200) {
        Serial.printf("[COMM ERR] GET /config HTTP %d\n", code);
        http.end();
        set_phase_code(0);
        return false;
    }

    set_phase_code(5); // Phase 5: RESPONSE WAIT / PARSE
    String body = http.getString();
    http.end();
    set_phase_code(0);

    int idx;
    idx = body.indexOf("\"round\":");
    if (idx >= 0) round = body.substring(idx + 8).toInt();

    idx = body.indexOf("\"epochs\":");
    if (idx >= 0) epochs = body.substring(idx + 9).toInt();

    idx = body.indexOf("\"settle_delay_ms\":");
    if (idx >= 0) g_inter_phase_settle_ms = body.substring(idx + 18).toInt();

    idx = body.indexOf("\"assigned_client_id\":");
    if (idx >= 0) client_id = body.substring(idx + 21).toInt();

    idx = body.indexOf("\"is_finished\":");
    if (idx >= 0) {
        String sub = body.substring(idx + 14);
        sub.trim();
        is_finished = sub.startsWith("true");
    }

    return true;
}

bool download_global_model() {
    set_phase_code(2); // Phase 2: TCP CONNECT
    WiFiClient client;
    HTTPClient http;
    String url = String("http://") + SERVER_IP + ":" + SERVER_PORT + "/model";

    if (!http.begin(client, url)) {
        set_phase_code(0);
        return false;
    }

    set_phase_code(3); // Phase 3: HTTP REQUEST
    int code = http.GET();
    if (code != 200) {
        Serial.printf("[COMM ERR] GET /model HTTP %d\n", code);
        http.end();
        set_phase_code(0);
        return false;
    }

    set_phase_code(4); // Phase 4: HTTP PAYLOAD TRANSFER (3,120 bytes)
    WiFiClient *stream = http.getStreamPtr();
    size_t total = 0;
    unsigned long t0 = millis();

    while (http.connected() && total < FL_PARAM_BYTES && (millis() - t0 < 5000)) {
        size_t avail = stream->available();
        if (avail) {
            int read_bytes = stream->readBytes((char*)(g_model_buf + total), min(avail, (size_t)(FL_PARAM_BYTES - total)));
            total += read_bytes;
        }
        delay(1);
    }
    http.end();
    set_phase_code(0);

    if (total == FL_PARAM_BYTES) {
        memcpy(g_weights, g_model_buf, FL_NUM_WEIGHTS * sizeof(float));
        memcpy(g_biases, g_model_buf + FL_NUM_WEIGHTS * sizeof(float), FL_NUM_BIASES * sizeof(float));
        return true;
    }
    Serial.printf("[COMM ERR] Model download incomplete: got %d / %d bytes\n", total, FL_PARAM_BYTES);
    return false;
}

bool upload_trained_model(int client_id) {
    set_phase_code(2); // Phase 2: TCP CONNECT
    WiFiClient client;
    HTTPClient http;
    String url = String("http://") + SERVER_IP + ":" + SERVER_PORT + "/update";

    memcpy(g_model_buf, g_weights, FL_NUM_WEIGHTS * sizeof(float));
    memcpy(g_model_buf + FL_NUM_WEIGHTS * sizeof(float), g_biases, FL_NUM_BIASES * sizeof(float));

    if (!http.begin(client, url)) {
        set_phase_code(0);
        return false;
    }

    set_phase_code(3); // Phase 3: HTTP REQUEST / HEADERS
    http.addHeader("Content-Type", "application/octet-stream");
    http.addHeader("X-Client-ID", String(client_id));
    http.addHeader("X-Num-Samples", String(FL_NUM_SAMPLES));

    set_phase_code(4); // Phase 4: HTTP PAYLOAD TRANSFER
    int code = http.POST(g_model_buf, FL_PARAM_BYTES);

    set_phase_code(5); // Phase 5: SERVER RESPONSE WAIT / ACK
    http.end();
    set_phase_code(0);

    if (code == 200) {
        return true;
    }
    Serial.printf("[COMM ERR] POST /update HTTP %d\n", code);
    return false;
}

/* ===================================================================
 * Pull-Down Resistor Load Bias Calibration Routine
 * Measures baseline power with marker pins forced LOW vs. HIGH.
 * =================================================================== */
void run_pulldown_bias_calibration() {
    Serial.println("\n# [CALIBRATION] Starting 100 kOhm Pull-Down Bias Calibration...");
    
    // Step 1: Force all 5 marker lines LOW for 2,000 ms (idle baseline)
    digitalWrite(PIN_GPIO_TRAIN, LOW);
    digitalWrite(PIN_GPIO_COMM,  LOW);
    digitalWrite(PIN_PHASE_BIT0, LOW);
    digitalWrite(PIN_PHASE_BIT1, LOW);
    digitalWrite(PIN_PHASE_BIT2, LOW);
    Serial.println("# [CALIBRATION] Step 1: All markers LOW (2000 ms)...");
    delay(2000);

    // Step 2: Force all 5 marker lines HIGH for 2,000 ms without CPU load
    // (Loads all 5 pull-down resistors at 3.3V)
    digitalWrite(PIN_GPIO_TRAIN, HIGH);
    digitalWrite(PIN_GPIO_COMM,  HIGH);
    digitalWrite(PIN_PHASE_BIT0, HIGH);
    digitalWrite(PIN_PHASE_BIT1, HIGH);
    digitalWrite(PIN_PHASE_BIT2, HIGH);
    Serial.println("# [CALIBRATION] Step 2: All markers HIGH (2000 ms, measuring pull-down load)...");
    delay(2000);

    // Step 3: Return all marker lines to LOW
    digitalWrite(PIN_GPIO_TRAIN, LOW);
    digitalWrite(PIN_GPIO_COMM,  LOW);
    digitalWrite(PIN_PHASE_BIT0, LOW);
    digitalWrite(PIN_PHASE_BIT1, LOW);
    digitalWrite(PIN_PHASE_BIT2, LOW);
    Serial.println("# [CALIBRATION] Step 3: All markers LOW (2000 ms). Calibration complete.\n");
    delay(2000);
}

/* ===================================================================
 * Pre-Run Clock-Sync Strobe (Executed ONCE in setup())
 * Excluded from energy integration; marked with distinct Phase Code 7.
 * =================================================================== */
void emit_pre_run_clock_sync() {
    Serial.println("# [SYNC] Transmitting single pre-run calibration sync strobe (Phase 7)...");
    
    // 1. Assert Phase Code 7 (111) for 2 ms
    set_phase_code(7);

    // 2. Transmit distinctive UDP sync datagram with magic to host port 9999
    uint32_t sync_packet[3];
    sync_packet[0] = 0xDEADBEEF;
    sync_packet[1] = 0; // Pre-run run indicator
    sync_packet[2] = (uint32_t)micros();

    udp_sync.beginPacket(SERVER_IP, SYNC_UDP_PORT);
    udp_sync.write((const uint8_t*)sync_packet, sizeof(sync_packet));
    udp_sync.endPacket();

    delay(2); // 2 ms sharp strobe
    set_phase_code(0); // Return to IDLE

    Serial.println("# [SYNC] Pre-run clock sync completed. Settling for 200 ms before FL rounds.");
    delay(200);
}

/* ===================================================================
 * Setup & Main Loop
 * =================================================================== */
void setup() {
    Serial.begin(115200);
    delay(1000);

    pinMode(PIN_GPIO_TRAIN, OUTPUT);
    pinMode(PIN_GPIO_COMM, OUTPUT);
    pinMode(PIN_PHASE_BIT0, OUTPUT);
    pinMode(PIN_PHASE_BIT1, OUTPUT);
    pinMode(PIN_PHASE_BIT2, OUTPUT);

    set_phase_code(0);

    Serial.println("\n# ===================================================");
    Serial.println("# Unified ESP32-S3 FL Client (Board A DUT) Started");
    Serial.printf("# Sweep Support: E in {1, 2, 5, 8, 10, 20}\n");
    Serial.printf("# Clean Tail Isolation Mode: %s (%d ms delay)\n",
                  CLEAN_TAIL_MEASUREMENT_MODE ? "ENABLED" : "DISABLED", INTER_PHASE_SETTLE_MS);
    Serial.printf("# Active Sleep Mode: %d (0=Standby, 1=Modem, 2=Light, 3=Deep)\n", ACTIVE_SLEEP_MODE);
    Serial.println("# Phase Codes: 0=IDLE, 1=DNS, 2=TCP, 3=HTTP_REQ, 4=PAYLOAD, 5=WAIT, 6=TRAIN, 7=SYNC");
    Serial.println("# NOTE: External 100 kOhm pull-down resistors required on GPIO 10-14");
    Serial.println("# ===================================================");

    #if ENABLE_PULLDOWN_BIAS_CALIBRATION
    run_pulldown_bias_calibration();
    #endif

    for (int i = 0; i < FL_NUM_WEIGHTS; i++) g_weights[i] = 0.01f * (float)(i % 7 - 3);
    for (int i = 0; i < FL_NUM_BIASES; i++)  g_biases[i] = 0.0f;

    Serial.printf("# Connecting to Wi-Fi SSID: %s\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    if (ACTIVE_SLEEP_MODE == SLEEP_ACTIVE_STANDBY) {
        WiFi.setSleep(false);
    } else {
        WiFi.setSleep(true);
    }
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\n# [INFO] Wi-Fi Connected! IP: " + WiFi.localIP().toString());

    // Fired ONCE before benchmark window begins; completely excluded from FL round energy
    emit_pre_run_clock_sync();
    set_phase_code(0);
    Serial.println("# [DUT_READY] Waiting for host trigger command CMD:RUN_TURN...");
}

void execute_client_turn(int epochs, int delay_ms) {
    int round = 0;
    int client_id = 0;
    bool is_finished = false;
    unsigned long turn_start_ms = millis();

    Serial.printf("\n# >>> Starting Turn | Epochs E=%d | Delay=%d ms <<<\n", epochs, delay_ms);

    // 1. Fetch Server Config (instrumented with phase codes 2, 3, 5)
    bool config_ok = fetch_server_config(round, epochs, client_id, is_finished);
    if (!config_ok) {
        Serial.println("# [COMM ERR] fetch_server_config failed");
        set_phase_code(0);
        return;
    }

    #if CLEAN_TAIL_MEASUREMENT_MODE
    if (g_inter_phase_settle_ms > 0) delay(g_inter_phase_settle_ms);
    #endif

    // 2. Download Global Model (instrumented with phase codes 2, 3, 4)
    download_global_model();

    #if CLEAN_TAIL_MEASUREMENT_MODE
    // Clean settling delay between GET /model falling edge and TRAIN rising edge
    if (g_inter_phase_settle_ms > 0) delay(g_inter_phase_settle_ms);
    #endif

    // 3. Local SGD Training (instrumented with phase code 6)
    unsigned long t_train_0 = millis();
    train_local_model(epochs, client_id);
    unsigned long train_dur = millis() - t_train_0;
    Serial.printf("# [TRAIN COMPLETE] Duration: %lu ms\n", train_dur);

    #if CLEAN_TAIL_MEASUREMENT_MODE
    // Clean settling delay between TRAIN falling edge and POST /update rising edge
    if (g_inter_phase_settle_ms > 0) delay(g_inter_phase_settle_ms);
    #endif

    // 4. Upload Trained Model (instrumented with phase codes 2, 3, 4, 5)
    unsigned long t_comm_0 = millis();
    upload_trained_model(client_id);
    unsigned long comm_dur = millis() - t_comm_0;
    Serial.printf("# [UPLOAD COMPLETE] Duration: %lu ms\n", comm_dur);

    #if CLEAN_TAIL_MEASUREMENT_MODE
    // Clean settling delay after POST /update falling edge
    if (g_inter_phase_settle_ms > 0) delay(g_inter_phase_settle_ms);
    #endif

    set_phase_code(0);
    Serial.println("# [DUT_TURN_DONE]");
}

void loop() {
    // Maintain strict baseline state: all markers LOW
    set_phase_code(0);

    if (Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd.startsWith("CMD:RUN_TURN")) {
            // Expected syntax: "CMD:RUN_TURN <epochs> <settle_delay_ms>"
            int epochs = 5;
            int delay_ms = 60;
            int space1 = cmd.indexOf(' ');
            if (space1 > 0) {
                int space2 = cmd.indexOf(' ', space1 + 1);
                if (space2 > 0) {
                    epochs = cmd.substring(space1 + 1, space2).toInt();
                    delay_ms = cmd.substring(space2 + 1).toInt();
                } else {
                    epochs = cmd.substring(space1 + 1).toInt();
                }
            }
            g_inter_phase_settle_ms = delay_ms;
            execute_client_turn(epochs, delay_ms);
        } else if (cmd.startsWith("CMD:PING")) {
            Serial.println("# [PONG]");
        }
    }
    delay(5);
}
