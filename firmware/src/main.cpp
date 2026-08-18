#include <Arduino.h>
#include "mpu6050_driver.h"
#include "sampling_timer.h"
#include "serial_protocol.h"

void setup() {
    setupSerial();

    if (!setupMPU6050()) {
        writeErrorMarker("INIT_FAILED");
    }

    setupSamplingTimer();
}

void loop() {
    if (!sampleReadyFlag) {
        return;
    }
    sampleReadyFlag = false;

    RawSample sample;
    if (!readMPU6050Raw(sample)) {
        writeErrorMarker("I2C_READ_FAILED");
        return;
    }

    writeSampleCsv(sample);
}
