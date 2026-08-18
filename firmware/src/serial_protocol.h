#ifndef SERIAL_PROTOCOL_H
#define SERIAL_PROTOCOL_H

#include "mpu6050_driver.h"
#include <Arduino.h>

constexpr uint32_t SERIAL_BAUD_RATE = 115200;

void setupSerial();
void writeSampleCsv(const RawSample& sample);
void writeErrorMarker(const char* reason);

#endif // SERIAL_PROTOCOL_H
