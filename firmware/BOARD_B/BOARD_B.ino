/**
 * @file BOARD_B.ino
 * @brief Unified ESP32-S3 High-Speed Energy Logger Host Firmware
 *
 * Hardware:
 *   - Board B: ESP32-S3 Dev Board
 *   - Current Sensor: TI INA238 breakout with R015 (15 mOhm, 1%) shunt
 *
 * Pinout Connections:
 *   - Board B GPIO 4  --> INA238 SDA
 *   - Board B GPIO 5  --> INA238 SCL
 *   - Board B 3V3     --> INA238 VCC
 *   - Board B GND     --> INA238 GND (and Common GND)
 *   - Board B GPIO 6  <-- Board A (DUT) GPIO 10 (TRAIN Strobe)
 *   - Board B GPIO 7  <-- Board A (DUT) GPIO 11 (COMM Strobe)
 *   - Board B GPIO 8  <-- Board A (DUT) GPIO 12 (Sub-phase Bit 0)
 *   - Board B GPIO 9  <-- Board A (DUT) GPIO 13 (Sub-phase Bit 1)
 *   - Board B GPIO 18 <-- Board A (DUT) GPIO 14 (Sub-phase Bit 2)
 *
 * CRITICAL HARDWARE REQUIREMENT:
 *   - 100 kOhm external pull-down resistors MUST be connected between each marker
 *     line (GPIO 6, 7, 8, 9, 18) and GND (replaces legacy 10 kOhm spec to eliminate
 *     the ~1.3 mA / ~4.4 mW pull-down load bias during training).
 *
 * Transition Back-Dating & Raw Code Storage:
 *   - Maintains a 1-sample pipeline delay to confirm candidate transitions.
 *   - When a transition is confirmed stable at sample t, sample t-1 is back-dated
 *     to the first sample of the new code, preventing phase edge shifts or erasure.
 *   - Emits BOTH raw_phase_code and debounced_phase_code in a 9-column CSV schema.
 *
 * Output CSV Format:
 *   timestamp_ms,voltage_V,current_A,power_mW,gpio_train,gpio_comm,raw_phase_code,debounced_phase_code,dropped_samples
 */

#include <Arduino.h>
#include <Wire.h>

/* ===================================================================
 * Pin Definitions
 * =================================================================== */
#define PIN_I2C_SDA         4
#define PIN_I2C_SCL         5
#define PIN_GPIO_TRAIN      6   // Input: Connected to Board A GPIO 10 (Training)
#define PIN_GPIO_COMM       7   // Input: Connected to Board A GPIO 11 (Communication)
#define PIN_PHASE_BIT0      8   // Input: Connected to Board A GPIO 12 (Bit 0)
#define PIN_PHASE_BIT1      9   // Input: Connected to Board A GPIO 13 (Bit 1)
#define PIN_PHASE_BIT2      18  // Input: Connected to Board A GPIO 14 (Bit 2)

/* ===================================================================
 * Sampling Rate Configuration (500 Hz or 1000 Hz)
 * =================================================================== */
#define SAMPLE_RATE_HZ      1000
#define SAMPLE_INTERVAL_US  (1000000UL / SAMPLE_RATE_HZ)

/* ===================================================================
 * TI INA238 Minimal Driver
 * =================================================================== */
#define INA238_ADDR             0x40
#define INA238_REG_ADC_CONFIG   0x01
#define INA238_REG_VSHUNT       0x04
#define INA238_REG_VBUS         0x05
#define INA238_REG_DEVICE_ID    0x3F

#define INA238_SHUNT_RESISTOR_OHM   0.015f      // R015 shunt (15 mOhm)
#define INA238_VBUS_LSB_V           0.003125f   // 3.125 mV per LSB
#define INA238_VSHUNT_LSB_V         0.000005f   // 5.0 uV per LSB

class MiniINA238 {
public:
    bool begin(TwoWire &wire = Wire) {
        _wire = &wire;
        uint16_t dev_id = readReg(INA238_REG_DEVICE_ID);
        if ((dev_id & 0xFFF0) != 0x2380 && dev_id != 0x2381) {
            Serial.printf("# [WARN] INA238 Device ID: 0x%04X (expected 0x2380)\n", dev_id);
        }

        #if SAMPLE_RATE_HZ == 1000
        // Continuous Shunt & Bus (0xF), 280 us conversion time (0x3), 1x averaging (0x0)
        uint16_t adc_config = (0xF << 12) | (0x3 << 9) | (0x3 << 6) | (0x0 << 3) | 0x0;
        #else
        // Continuous Shunt & Bus (0xF), 540 us conversion time (0x4), 1x averaging (0x0)
        uint16_t adc_config = (0xF << 12) | (0x4 << 9) | (0x4 << 6) | (0x0 << 3) | 0x0;
        #endif

        writeReg(INA238_REG_ADC_CONFIG, adc_config);
        return true;
    }

    inline float readBusVoltage() {
        uint16_t raw = readReg(INA238_REG_VBUS);
        return ((float)raw) * INA238_VBUS_LSB_V;
    }

    inline float readShuntVoltage() {
        int16_t raw = (int16_t)readReg(INA238_REG_VSHUNT);
        return ((float)raw) * INA238_VSHUNT_LSB_V;
    }

    inline float readCurrent() {
        return readShuntVoltage() / INA238_SHUNT_RESISTOR_OHM;
    }

    inline float readPower_mW(float vbus, float current) {
        return vbus * current * 1000.0f;
    }

private:
    TwoWire *_wire;

    void writeReg(uint8_t reg, uint16_t val) {
        _wire->beginTransmission(INA238_ADDR);
        _wire->write(reg);
        _wire->write((uint8_t)(val >> 8));
        _wire->write((uint8_t)(val & 0xFF));
        _wire->endTransmission();
    }

    inline uint16_t readReg(uint8_t reg) {
        _wire->beginTransmission(INA238_ADDR);
        _wire->write(reg);
        _wire->endTransmission(false);
        _wire->requestFrom((int)INA238_ADDR, 2);
        if (_wire->available() >= 2) {
            uint8_t msb = _wire->read();
            uint8_t lsb = _wire->read();
            return (uint16_t)((msb << 8) | lsb);
        }
        return 0;
    }
};

MiniINA238 ina238;
bool is_logging = true;
unsigned long next_sample_time_us = 0;
unsigned long start_time_ms = 0;
unsigned long dropped_samples_total = 0;

// Sample pipeline buffer for 1-sample back-dating
struct SampleBuffer {
    unsigned long timestamp_ms;
    float vbus;
    float current;
    float power;
    int gpio_train;
    int gpio_comm;
    uint8_t raw_code;
    uint8_t debounced_code;
    unsigned long dropped_samples;
    bool valid;
};

static SampleBuffer prev_sample = {0, 0, 0, 0, 0, 0, 0, 0, 0, false};
static uint8_t current_stable_code = 0;

void setup() {
    Serial.begin(921600);
    delay(1000);

    Serial.println("\n# ===================================================");
    Serial.println("# Unified ESP32-S3 High-Speed Energy Logger (Board B)");
    Serial.printf("# Sampling Rate: %d Hz (Interval: %lu us)\n", SAMPLE_RATE_HZ, (unsigned long)SAMPLE_INTERVAL_US);
    Serial.println("# Baud Rate: 921,600 (Fast USB-CDC Stream)");
    Serial.println("# Pipeline: Back-dated debounce with raw code capture");
    Serial.println("# Hardware: 10 kOhm external pull-downs on GPIO 6, 7, 8, 9, 18");
    Serial.println("# ===================================================");

    pinMode(PIN_GPIO_TRAIN, INPUT_PULLDOWN);
    pinMode(PIN_GPIO_COMM, INPUT_PULLDOWN);
    pinMode(PIN_PHASE_BIT0, INPUT_PULLDOWN);
    pinMode(PIN_PHASE_BIT1, INPUT_PULLDOWN);
    pinMode(PIN_PHASE_BIT2, INPUT_PULLDOWN);

    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000); // 400 kHz Fast-Mode I2C

    if (ina238.begin(Wire)) {
        Serial.println("# [INFO] INA238 initialized successfully.");
    } else {
        Serial.println("# [ERROR] INA238 I2C communication failed! Check wiring.");
    }

    // Noise-Floor Calibration Routine (50 baseline samples)
    float sum_v = 0.0f, sum_i = 0.0f, sum_i_sq = 0.0f;
    const int CAL_SAMPLES = 50;
    for (int i = 0; i < CAL_SAMPLES; i++) {
        float v = ina238.readBusVoltage();
        float curr = ina238.readCurrent();
        sum_v += v;
        sum_i += curr;
        sum_i_sq += curr * curr;
        delayMicroseconds(SAMPLE_INTERVAL_US);
    }
    float mean_v = sum_v / CAL_SAMPLES;
    float mean_i = sum_i / CAL_SAMPLES;
    float rms_noise_i = sqrtf(max(0.0f, (sum_i_sq / CAL_SAMPLES) - (mean_i * mean_i)));

    Serial.printf("# [CALIBRATION] Mean Vbus: %.3f V | Baseline Current: %.4f A (%.1f mW)\n",
                  mean_v, mean_i, mean_v * mean_i * 1000.0f);
    Serial.printf("# [CALIBRATION] RMS Current Noise Floor: %.3f mA\n", rms_noise_i * 1000.0f);
    Serial.println("# CSV Header: timestamp_ms,voltage_V,current_A,power_mW,gpio_train,gpio_comm,raw_phase_code,debounced_phase_code,dropped_samples");

    start_time_ms = millis();
    next_sample_time_us = micros() + SAMPLE_INTERVAL_US;
}

void loop() {
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd.equalsIgnoreCase("START")) {
            is_logging = true;
            start_time_ms = millis();
            next_sample_time_us = micros();
            dropped_samples_total = 0;
            prev_sample.valid = false;
            current_stable_code = 0;
            Serial.println("# [CMD] LOGGING_STARTED");
        } else if (cmd.equalsIgnoreCase("STOP")) {
            is_logging = false;
            // Flush last buffered sample if valid
            if (prev_sample.valid) {
                Serial.printf("%lu,%.4f,%.6f,%.3f,%d,%d,%u,%u,%lu\n",
                              prev_sample.timestamp_ms, prev_sample.vbus,
                              prev_sample.current, prev_sample.power,
                              prev_sample.gpio_train, prev_sample.gpio_comm,
                              prev_sample.raw_code, prev_sample.debounced_code,
                              prev_sample.dropped_samples);
                prev_sample.valid = false;
            }
            Serial.printf("# [CMD] LOGGING_STOPPED | Total Dropped Samples: %lu\n", dropped_samples_total);
        }
    }

    if (!is_logging) {
        delay(10);
        return;
    }

    unsigned long current_time_us = micros();
    if ((long)(current_time_us - next_sample_time_us) >= 0) {
        if ((long)(current_time_us - next_sample_time_us) > (long)(SAMPLE_INTERVAL_US)) {
            unsigned long missed = (current_time_us - next_sample_time_us) / SAMPLE_INTERVAL_US;
            dropped_samples_total += missed;
            next_sample_time_us += (missed + 1) * SAMPLE_INTERVAL_US;
        } else {
            next_sample_time_us += SAMPLE_INTERVAL_US;
        }

        int gpio_train = digitalRead(PIN_GPIO_TRAIN);
        int gpio_comm  = digitalRead(PIN_GPIO_COMM);

        // Read 3-bit raw phase code
        uint8_t raw_code = 0;
        if (digitalRead(PIN_PHASE_BIT0)) raw_code |= 1;
        if (digitalRead(PIN_PHASE_BIT1)) raw_code |= 2;
        if (digitalRead(PIN_PHASE_BIT2)) raw_code |= 4;

        float vbus = ina238.readBusVoltage();
        float current = ina238.readCurrent();
        float power = ina238.readPower_mW(vbus, current);
        unsigned long now_ms = millis() - start_time_ms;

        // Back-dating logic with 1-sample pipeline:
        // If prev_sample is valid, evaluate whether it transitioned:
        if (prev_sample.valid) {
            if (raw_code == prev_sample.raw_code) {
                // The transition seen at prev_sample is confirmed by this current sample!
                // Back-date prev_sample's debounced_code to raw_code!
                prev_sample.debounced_code = prev_sample.raw_code;
                current_stable_code = prev_sample.raw_code;
            } else {
                // Mismatch: prev_sample was a transient spike, or this is a transition in progress.
                // Keep debounced_code at current_stable_code.
                prev_sample.debounced_code = current_stable_code;
            }

            // Emit the back-dated previous sample
            Serial.printf("%lu,%.4f,%.6f,%.3f,%d,%d,%u,%u,%lu\n",
                          prev_sample.timestamp_ms, prev_sample.vbus,
                          prev_sample.current, prev_sample.power,
                          prev_sample.gpio_train, prev_sample.gpio_comm,
                          prev_sample.raw_code, prev_sample.debounced_code,
                          prev_sample.dropped_samples);
        }

        // Store current sample into buffer
        prev_sample.timestamp_ms = now_ms;
        prev_sample.vbus = vbus;
        prev_sample.current = current;
        prev_sample.power = power;
        prev_sample.gpio_train = gpio_train;
        prev_sample.gpio_comm = gpio_comm;
        prev_sample.raw_code = raw_code;
        prev_sample.debounced_code = current_stable_code; // Tentative
        prev_sample.dropped_samples = dropped_samples_total;
        prev_sample.valid = true;
    }
}
