#include "sampling_timer.h"

hw_timer_t* samplingTimer = nullptr;
volatile bool sampleReadyFlag = false;

void IRAM_ATTR onSampleTimer() {
    sampleReadyFlag = true;
}

void setupSamplingTimer() {
    // Timer 0, prescaler 80 -> 1 MHz count frequency (1 tick = 1 us)
    samplingTimer = timerBegin(0, 80, true);
    timerAttachInterrupt(samplingTimer, &onSampleTimer, true);
    timerAlarmWrite(samplingTimer, SAMPLE_INTERVAL_US, true);  // 10000 us = 100 Hz
    timerAlarmEnable(samplingTimer);
}
