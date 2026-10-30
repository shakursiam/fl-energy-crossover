/**
 * @file BOARD_A_SLEEP.ino
 * @brief Campaign 4: board-level sleep power and wake-up energy of the ESP32-S3 DUT
 *
 * Same hardware and wiring as BOARD_A.ino (see docs/wiring.md, fig_wiring.pdf):
 *   - Board A GPIO 10 --> Board B GPIO 6  (TRAIN, used only by the boot self-test)
 *   - Board A GPIO 11 --> Board B GPIO 7  (COMM,  used only by the boot self-test)
 *   - Board A GPIO 12 --> Board B GPIO 8  (phase bit 0)
 *   - Board A GPIO 13 --> Board B GPIO 9  (phase bit 1)
 *   - Board A GPIO 14 --> Board B GPIO 18 (phase bit 2)
 *   - 100 kOhm pull-downs on all five lines; DUT powered at 5 V through the INA238 shunt.
 *
 * Every cycle measures, in this order:
 *   1. Active standby  (Wi-Fi associated, power save off)   -- the P_base condition
 *   2. Modem sleep     (Wi-Fi associated, WIFI_PS_MIN_MODEM)
 *   3. Light sleep     (forced, timer wake-up; Wi-Fi stopped before and re-associated after)
 *   4. Deep sleep      (timer wake-up; cold boot, Wi-Fi association and DHCP after)
 *
 * Phase codes on the 3-bit bus (marker lines are LOW inside every measured window, so the
 * pull-down resistors add no load to the sleep measurements):
 *   1 = standby window delimiter   (20 ms pulse at start and end)
 *   2 = modem-sleep window delimiter
 *   3 = light-sleep delimiter      (pulse just before sleep and just after wake-up)
 *   4 = deep-sleep entry           (pulse just before esp_deep_sleep_start)
 *   5 = HELD from the first instruction of setup() after a deep-sleep wake-up until an IP
 *       address is obtained (deep-sleep wake-up window)
 *   6 = HELD from light-sleep wake-up until Wi-Fi is re-associated with an IP address
 *   7 = session markers / self-test / error flags
 *
 * Cycle state survives deep sleep in RTC memory. After N_CYCLES the DUT stays awake in standby.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_sleep.h>
#include <esp_wifi.h>

#define PIN_GPIO_TRAIN     10
#define PIN_GPIO_COMM      11
#define PIN_PHASE_BIT0     12
#define PIN_PHASE_BIT1     13
#define PIN_PHASE_BIT2     14

const char* WIFI_SSID = "YOUR_SSID";        // set before flashing
const char* WIFI_PASS = "YOUR_PASSWORD";    // set before flashing

#ifndef N_CYCLES
#define N_CYCLES          20
#endif
#ifndef T_STATE_MS
#define T_STATE_MS        30000UL   // length of every measured state
#endif
#define T_GAP_MS          3000UL    // awake settling before each state
#define PULSE_MS          20
#define WIFI_TIMEOUT_MS   30000UL

enum PhaseCode : uint8_t {
    C_IDLE = 0, C_STANDBY = 1, C_MODEM = 2, C_LIGHT = 3, C_DEEP = 4,
    C_WAKE_DEEP = 5, C_WAKE_LIGHT = 6, C_SESSION = 7
};

RTC_DATA_ATTR int32_t  g_cycle = -1;      // -1 = fresh power-on / reset
RTC_DATA_ATTR uint32_t g_boot_count = 0;

static inline void mark(uint8_t code) {
    digitalWrite(PIN_PHASE_BIT0, (code & 1) ? HIGH : LOW);
    digitalWrite(PIN_PHASE_BIT1, (code & 2) ? HIGH : LOW);
    digitalWrite(PIN_PHASE_BIT2, (code & 4) ? HIGH : LOW);
}

static void pulse(uint8_t code, uint32_t ms = PULSE_MS) {
    mark(code);
    delay(ms);
    mark(C_IDLE);
}

static void init_pins() {
    const int pins[] = {PIN_GPIO_TRAIN, PIN_GPIO_COMM, PIN_PHASE_BIT0, PIN_PHASE_BIT1, PIN_PHASE_BIT2};
    for (int p : pins) {
        pinMode(p, OUTPUT);
        digitalWrite(p, LOW);
    }
}

// Raise each marker line alone for 300 ms so the logger can verify all five wires.
static void marker_self_test() {
    const int pins[] = {PIN_GPIO_TRAIN, PIN_GPIO_COMM, PIN_PHASE_BIT0, PIN_PHASE_BIT1, PIN_PHASE_BIT2};
    for (int p : pins) {
        digitalWrite(p, HIGH);
        delay(300);
        digitalWrite(p, LOW);
        delay(300);
    }
}

// WiFi.setSleep() in arduino-esp32 3.3.6 applies the previously stored mode (it calls
// esp_wifi_set_ps(_sleepEnabled) before updating it), so the IDF call is used directly.
static wifi_ps_type_t set_ps(wifi_ps_type_t mode) {
    esp_wifi_set_ps(mode);
    wifi_ps_type_t now = WIFI_PS_NONE;
    esp_wifi_get_ps(&now);
    return now;
}

// Associates with the AP (power save off) and waits for DHCP. Returns elapsed ms, or -1 on failure.
static long wifi_connect() {
    unsigned long t0 = millis();
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);   // stored and applied by the core on connection
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - t0 > WIFI_TIMEOUT_MS) return -1;
        delay(5);
    }
    set_ps(WIFI_PS_NONE);
    return (long)(millis() - t0);
}

static void wifi_off() {
    WiFi.disconnect(true);
    WiFi.mode(WIFI_OFF);
}

static void error_flag() {
    for (int i = 0; i < 3; i++) { pulse(C_SESSION, 100); delay(100); }
}

// States 1-4. Never returns for the deep-sleep state.
static void run_cycle() {
    Serial.printf("# [CYCLE %ld/%d] standby\n", (long)g_cycle + 1, N_CYCLES);

    // 1. Active standby: identical to the P_base condition of the FL campaigns.
    wifi_ps_type_t ps_standby = set_ps(WIFI_PS_NONE);
    delay(T_GAP_MS);
    pulse(C_STANDBY);
    delay(T_STATE_MS);
    pulse(C_STANDBY);

    // 2. Modem sleep: association kept, radio off between DTIM beacons, CPU idle at 240 MHz.
    wifi_ps_type_t ps_modem = set_ps(WIFI_PS_MIN_MODEM);
    delay(T_GAP_MS);
    pulse(C_MODEM);
    delay(T_STATE_MS);
    pulse(C_MODEM);
    set_ps(WIFI_PS_NONE);
    Serial.printf("# [CYCLE %ld] ps_standby=%d ps_modem=%d connected=%d RSSI=%d dBm\n", (long)g_cycle + 1,
                  (int)ps_standby, (int)ps_modem, (int)WiFi.isConnected(), WiFi.RSSI());

    // 3. Light sleep (forced). Wi-Fi is stopped first; association is re-established on wake-up.
    delay(T_GAP_MS);
    wifi_off();
    Serial.flush();
    pulse(C_LIGHT);
    esp_sleep_enable_timer_wakeup((uint64_t)T_STATE_MS * 1000ULL);
    esp_light_sleep_start();
    pulse(C_LIGHT);
    mark(C_WAKE_LIGHT);
    long t_light = wifi_connect();
    mark(C_IDLE);
    Serial.printf("# [CYCLE %ld] light wake: reconnect_ms=%ld\n", (long)g_cycle + 1, t_light);
    if (t_light < 0) error_flag();

    // 4. Deep sleep. Wake-up continues in setup().
    delay(T_GAP_MS);
    Serial.printf("# [CYCLE %ld] entering deep sleep\n", (long)g_cycle + 1);
    Serial.flush();
    wifi_off();
    pulse(C_DEEP);
    esp_sleep_enable_timer_wakeup((uint64_t)T_STATE_MS * 1000ULL);
    esp_deep_sleep_start();
}

static void finish_session() {
    Serial.println("# [SESSION] all cycles done; staying in active standby");
    pulse(C_SESSION, 500);
    g_cycle = -1;
}

void setup() {
    // Deep-sleep wake-up path: assert the wake marker before anything else.
    bool deep_wake = (esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_TIMER) && g_cycle >= 0;
    if (deep_wake) {
        pinMode(PIN_PHASE_BIT0, OUTPUT);
        pinMode(PIN_PHASE_BIT1, OUTPUT);
        pinMode(PIN_PHASE_BIT2, OUTPUT);
        mark(C_WAKE_DEEP);
    }
    g_boot_count++;
    Serial.begin(115200);
    Serial.setTxTimeoutMs(0);   // never block on USB when no host is reading
    init_pins();
    if (deep_wake) mark(C_WAKE_DEEP);  // init_pins() cleared the bits

    if (deep_wake) {
        long t_deep = wifi_connect();
        mark(C_IDLE);
        Serial.printf("# [CYCLE %ld] deep wake: boot_count=%lu reconnect_ms=%ld (since app start)\n",
                      (long)g_cycle + 1, (unsigned long)g_boot_count, t_deep);
        if (t_deep < 0) error_flag();
        g_cycle++;
        if (g_cycle >= N_CYCLES) {
            finish_session();
            return;
        }
        run_cycle();
        return;
    }

    // Fresh power-on or reset: self-test, then start the session.
    delay(1500);
    Serial.println("\n# ===================================================");
    Serial.println("# Board A: Campaign 4 sleep-power / wake-energy firmware");
    Serial.printf("# Cycles=%d  T_state=%lu ms  T_gap=%lu ms  reset_reason=%d\n",
                  N_CYCLES, T_STATE_MS, T_GAP_MS, (int)esp_reset_reason());
    Serial.println("# Codes: 1=standby 2=modem 3=light 4=deep 5=deep-wake 6=light-wake 7=session");
    Serial.println("# ===================================================");
    marker_self_test();
    long t0 = wifi_connect();
    Serial.printf("# [INFO] Wi-Fi connected in %ld ms, IP=%s\n", t0, WiFi.localIP().toString().c_str());
    if (t0 < 0) error_flag();
    delay(2000);
    pulse(C_SESSION, 500);
    g_cycle = 0;
    run_cycle();
}

void loop() {
    delay(5);
}
