#ifndef SAMPLING_TIMER_H
#define SAMPLING_TIMER_H

#include <Arduino.h>
#include <cstdint>

constexpr uint32_t SAMPLE_RATE_HZ = 100;
constexpr uint32_t SAMPLE_INTERVAL_US = 1000000 / SAMPLE_RATE_HZ;  // 10,000 us

extern volatile bool sampleReadyFlag;

void setupSamplingTimer();

#endif // SAMPLING_TIMER_H
