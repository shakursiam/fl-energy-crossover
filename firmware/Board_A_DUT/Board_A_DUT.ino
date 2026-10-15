/**
 * @file Board_A_DUT.ino
 * @brief ESP32-S3 Federated Learning Client — Device Under Test (Arduino IDE)
 * 
 * Hardware:
 *   - Board A: ESP32-S3 Dev Board (Powered through INA238 Shunt)
 * 
 * Pinout Connections:
 *   - Board A GPIO 10 (Output) --> Board B GPIO 6 (TRAIN Strobe)
 *   - Board A GPIO 11 (Output) --> Board B GPIO 7 (COMM Strobe)
 *   - Board A GND              --> Common GND (Board B GND & INA238 GND)
 * 
 * Wi-Fi & Server Configuration:
 *   - Network: 2.4 GHz (ESP32-S3 does not support 5 GHz)
 */

#include <WiFi.h>
#include <HTTPClient.h>

/* ===================================================================
 * Wi-Fi & Server Credentials (Configured for Photonics Lab bench)
 * =================================================================== */
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASS     = "YOUR_WIFI_PASSWORD";
const char* SERVER_IP     = "172.20.10.3";
const uint16_t SERVER_PORT = 8080;

/* ===================================================================
 * GPIO Phase Marker Pins (Cross-wired to Board B Logger)
 * =================================================================== */
#define PIN_GPIO_TRAIN    10   // HIGH during local SGD training
#define PIN_GPIO_COMM     11   // HIGH during Wi-Fi download/upload

/* ===================================================================
 * Model Dimensions & FL Hyperparameters
 * =================================================================== */
#define FL_NUM_INPUTS     64                                      // Backbone output features
#define FL_NUM_CLASSES    12                                      // KWS classes
#define FL_NUM_WEIGHTS    (FL_NUM_INPUTS * FL_NUM_CLASSES)        // 768 float params
#define FL_NUM_BIASES     FL_NUM_CLASSES                          // 12 float params
#define FL_NUM_PARAMS     (FL_NUM_WEIGHTS + FL_NUM_BIASES)        // 780 params total
#define FL_PARAM_BYTES    (FL_NUM_PARAMS * sizeof(float))         // 3,120 bytes

#define FL_BATCH_SIZE     16                                      // Mini-batch size
#define FL_LEARNING_RATE  0.01f                                   // SGD learning rate
#define FL_NUM_SAMPLES    100                                     // Samples per virtual client

/* ===================================================================
 * Global Model State & Mini-Batch Buffers
 * =================================================================== */
static float g_weights[FL_NUM_WEIGHTS];
static float g_biases[FL_NUM_BIASES];
static float g_grad_w[FL_NUM_WEIGHTS];
static float g_grad_b[FL_NUM_BIASES];

static float   g_features[FL_BATCH_SIZE * FL_NUM_INPUTS];
static float   g_logits[FL_BATCH_SIZE * FL_NUM_CLASSES];
static float   g_probs[FL_BATCH_SIZE * FL_NUM_CLASSES];
static uint8_t g_labels[FL_BATCH_SIZE];

static uint8_t g_model_buf[FL_PARAM_BYTES]; // 3,120 bytes binary buffer

/* Deterministic PRNG for synthetic dataset */
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

        // Stable softmax
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
        Serial.printf("  [TRAIN] Epoch %d/%d Loss: %.4f\n", e + 1, epochs, epoch_loss);
    }
}

/* ===================================================================
 * HTTP Communications with FL Server
 * =================================================================== */
bool fetch_server_config(int &round, int &epochs, int &client_id, bool &is_finished) {
    WiFiClient client;
    HTTPClient http;
    String url = String("http://") + SERVER_IP + ":" + SERVER_PORT + "/config";

    if (!http.begin(client, url)) return false;
    int code = http.GET();
    if (code != 200) {
        Serial.printf("[COMM ERR] GET /config HTTP %d\n", code);
        http.end();
        return false;
    }

    String body = http.getString();
    http.end();

    // Lightweight parsing of JSON fields
    int idx;
    idx = body.indexOf("\"round\":");
    if (idx >= 0) round = body.substring(idx + 8).toInt();

    idx = body.indexOf("\"epochs\":");
    if (idx >= 0) epochs = body.substring(idx + 9).toInt();

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
    WiFiClient client;
    HTTPClient http;
    String url = String("http://") + SERVER_IP + ":" + SERVER_PORT + "/model";

    if (!http.begin(client, url)) return false;
    int code = http.GET();
    if (code != 200) {
        Serial.printf("[COMM ERR] GET /model HTTP %d\n", code);
        http.end();
        return false;
    }

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

    if (total == FL_PARAM_BYTES) {
        memcpy(g_weights, g_model_buf, FL_NUM_WEIGHTS * sizeof(float));
        memcpy(g_biases, g_model_buf + FL_NUM_WEIGHTS * sizeof(float), FL_NUM_BIASES * sizeof(float));
        return true;
    }
    Serial.printf("[COMM ERR] Model download incomplete: got %d / %d bytes\n", total, FL_PARAM_BYTES);
    return false;
}

bool upload_trained_model(int client_id) {
    WiFiClient client;
    HTTPClient http;
    String url = String("http://") + SERVER_IP + ":" + SERVER_PORT + "/update";

    // Pack current weights and biases into binary buffer
    memcpy(g_model_buf, g_weights, FL_NUM_WEIGHTS * sizeof(float));
    memcpy(g_model_buf + FL_NUM_WEIGHTS * sizeof(float), g_biases, FL_NUM_BIASES * sizeof(float));

    if (!http.begin(client, url)) return false;
    http.addHeader("Content-Type", "application/octet-stream");
    http.addHeader("X-Client-ID", String(client_id));
    http.addHeader("X-Num-Samples", String(FL_NUM_SAMPLES));

    int code = http.POST(g_model_buf, FL_PARAM_BYTES);
    http.end();

    if (code == 200) {
        return true;
    }
    Serial.printf("[COMM ERR] POST /update HTTP %d\n", code);
    return false;
}

/* ===================================================================
 * Setup & Main Loop
 * =================================================================== */
void setup() {
    Serial.begin(115200);
    delay(1000);

    // Initialize GPIO Phase Strobes
    pinMode(PIN_GPIO_TRAIN, OUTPUT);
    pinMode(PIN_GPIO_COMM, OUTPUT);
    digitalWrite(PIN_GPIO_TRAIN, LOW);
    digitalWrite(PIN_GPIO_COMM, LOW);

    Serial.println("\n# =================================================");
    Serial.println("# ESP32-S3 Federated Learning DUT (Board A) Started");
    Serial.println("# =================================================");

    // Initialize model weights with small initial values
    for (int i = 0; i < FL_NUM_WEIGHTS; i++) g_weights[i] = 0.01f * (float)(i % 7 - 3);
    for (int i = 0; i < FL_NUM_BIASES; i++)  g_biases[i] = 0.0f;

    // Connect to 2.4 GHz Wi-Fi
    Serial.printf("# Connecting to Wi-Fi SSID: %s\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false); // Disable modem sleep for repeatable power benching
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\n# [INFO] Wi-Fi Connected! IP: " + WiFi.localIP().toString());
    Serial.println("# Ready for FL rounds from server: http://" + String(SERVER_IP) + ":" + String(SERVER_PORT));
}

void loop() {
    int round = 0;
    int epochs = 5;
    int client_id = 0;
    bool is_finished = false;

    // -------------------------------------------------------------
    // Query Server Config (No strobe - keep strobes strictly for model data)
    // -------------------------------------------------------------
    bool config_ok = fetch_server_config(round, epochs, client_id, is_finished);
    if (!config_ok) {
        delay(1000);
        return;
    }

    if (is_finished) {
        Serial.println("# [COMPLETED] FL Server signaled round finished. Waiting for next experiment...");
        delay(2000);
        return;
    }

    Serial.printf("\n# >>> Starting Round %d | Virtual Client ID: %d | Epochs: %d <<<\n", round, client_id, epochs);

    // -------------------------------------------------------------
    // Phase 1: Model Download (COMM HIGH)
    // -------------------------------------------------------------
    digitalWrite(PIN_GPIO_COMM, HIGH);
    download_global_model();
    digitalWrite(PIN_GPIO_COMM, LOW);

    // -------------------------------------------------------------
    // Phase 2: Local SGD Training (TRAIN HIGH)
    // -------------------------------------------------------------
    digitalWrite(PIN_GPIO_TRAIN, HIGH);
    unsigned long t_train_start = millis();
    train_local_model(epochs, client_id);
    unsigned long train_dur = millis() - t_train_start;
    digitalWrite(PIN_GPIO_TRAIN, LOW);
    Serial.printf("# [TRAIN COMPLETE] Duration: %lu ms\n", train_dur);

    // -------------------------------------------------------------
    // Phase 3: Model Upload (COMM HIGH)
    // -------------------------------------------------------------
    digitalWrite(PIN_GPIO_COMM, HIGH);
    unsigned long t_comm_start = millis();
    bool upload_ok = upload_trained_model(client_id);
    unsigned long comm_dur = millis() - t_comm_start;
    digitalWrite(PIN_GPIO_COMM, LOW);

    if (upload_ok) {
        Serial.printf("# [UPLOAD SUCCESS] Uploaded 3,120 B in %lu ms\n", comm_dur);
    }

    // Inter-round settling delay
    delay(500);
}
