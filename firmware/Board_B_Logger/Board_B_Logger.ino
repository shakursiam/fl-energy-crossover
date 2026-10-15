/**
 * @file Board_B_Logger.ino
 * @brief ESP32-S3 Energy Logging Host Firmware (Arduino IDE)
 * 
 * Hardware:
 *   - Board B: ESP32-S3 Dev Board
 *   - Current Sensor: TI INA238 breakout with R015 (15 mOhm) shunt
 * 
 * Pinout Connections:
 *   - Board B GPIO 4  --> INA238 SDA
 *   - Board B GPIO 5  --> INA238 SCL
 *   - Board B 3V3     --> INA238 VCC
 *   - Board B GND     --> INA238 GND (and Common GND)
 *   - Board B GPIO 6  <-- Board A (DUT) GPIO 10 (TRAIN Strobe)
 *   - Board B GPIO 7  <-- Board A (DUT) GPIO 11 (COMM Strobe)
 * 
 * Output:
 *   Streams CSV to Serial at 115200 baud (~45 Hz):
 *   timestamp_ms,voltage_V,current_A,power_mW,gpio_train,gpio_comm
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

/* High-Resolution Sampling Rate: 500 Hz (Interval = 2,000 us) */
#define SAMPLE_INTERVAL_US  2000UL

/* ===================================================================
 * Minimal Embedded INA238 Driver (No external library required)
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

        // ADC Config: Continuous Bus & Shunt, 540 us conversion time, 1x averaging (~1.08 ms hardware cycle)
        uint16_t adc_config = (0xF << 12) | (0x4 << 9) | (0x4 << 6) | (0x0 << 3) | 0x0;
        writeReg(INA238_REG_ADC_CONFIG, adc_config);
        return true;
    }

    float readBusVoltage() {
        uint16_t raw = readReg(INA238_REG_VBUS);
        return ((float)raw) * INA238_VBUS_LSB_V;
    }

    float readShuntVoltage() {
        int16_t raw = (int16_t)readReg(INA238_REG_VSHUNT);
        return ((float)raw) * INA238_VSHUNT_LSB_V;
    }

    float readCurrent() {
        return readShuntVoltage() / INA238_SHUNT_RESISTOR_OHM;
    }

    float readPower_mW(float vbus, float current) {
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

    uint16_t readReg(uint8_t reg) {
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

void setup() {
    Serial.begin(460800);
    delay(1000);

    Serial.println("\n# =================================================");
    Serial.println("# ESP32-S3 Energy Logger Host (Board B) Initializing");
    Serial.println("# =================================================");

    // Marker Inputs from DUT
    pinMode(PIN_GPIO_TRAIN, INPUT_PULLDOWN);
    pinMode(PIN_GPIO_COMM, INPUT_PULLDOWN);

    // Initialize I2C
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000);

    if (ina238.begin(Wire)) {
        Serial.println("# [INFO] INA238 initialized (R015 Shunt = 15 mOhm, 500 Hz high-resolution rate).");
    } else {
        Serial.println("# [ERROR] INA238 I2C communication failed! Check wiring.");
    }

    Serial.println("# CSV Header: timestamp_ms,voltage_V,current_A,power_mW,gpio_train,gpio_comm");
    
    start_time_ms = millis();
    next_sample_time_us = micros() + SAMPLE_INTERVAL_US;
}

void loop() {
    // Process serial control commands (START / STOP from PC runner)
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd.equalsIgnoreCase("START")) {
            is_logging = true;
            start_time_ms = millis();
            next_sample_time_us = micros();
            Serial.println("# [CMD] LOGGING_STARTED");
        } else if (cmd.equalsIgnoreCase("STOP")) {
            is_logging = false;
            Serial.println("# [CMD] LOGGING_STOPPED");
        }
    }

    if (!is_logging) {
        delay(10);
        return;
    }

    unsigned long current_time_us = micros();
    if ((long)(current_time_us - next_sample_time_us) >= 0) {
        next_sample_time_us += SAMPLE_INTERVAL_US;

        // Synchronous sample of markers and power monitor
        int gpio_train = digitalRead(PIN_GPIO_TRAIN);
        int gpio_comm  = digitalRead(PIN_GPIO_COMM);

        float vbus = ina238.readBusVoltage();
        float current = ina238.readCurrent();
        float power = ina238.readPower_mW(vbus, current);
        unsigned long now_ms = millis() - start_time_ms;

        // Emit CSV sample line
        Serial.printf("%lu,%.4f,%.6f,%.3f,%d,%d\n",
                      now_ms, vbus, current, power, gpio_train, gpio_comm);
    }
}
