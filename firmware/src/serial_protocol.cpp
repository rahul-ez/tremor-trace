#include "serial_protocol.h"
#include <cstdio>

void setupSerial() {
    Serial.begin(SERIAL_BAUD_RATE);
}

void writeSampleCsv(const RawSample& sample) {
    char line[96];
    int len = snprintf(line, sizeof(line), "%lu,%d,%d,%d,%d,%d,%d\n",
                       static_cast<unsigned long>(sample.timestamp_us),
                       sample.ax, sample.ay, sample.az,
                       sample.gx, sample.gy, sample.gz);
    if (len > 0 && len < static_cast<int>(sizeof(line))) {
        Serial.print(line);
    }
}

void writeErrorMarker(const char* reason) {
    char line[96];
    snprintf(line, sizeof(line), "ERR,%s\n", reason);
    Serial.print(line);
}
