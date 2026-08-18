#ifndef MPU6050_DRIVER_H
#define MPU6050_DRIVER_H

#include <Arduino.h>
#include <cstdint>

// I2C Configuration
constexpr uint8_t PIN_SDA = 21;
constexpr uint8_t PIN_SCL = 22;
constexpr uint32_t I2C_CLOCK_HZ = 400000;

// MPU6050 Register Map
constexpr uint8_t MPU6050_ADDR = 0x68;
constexpr uint8_t REG_SMPLRT_DIV = 0x19;
constexpr uint8_t REG_CONFIG = 0x1A;
constexpr uint8_t REG_GYRO_CONFIG = 0x1B;
constexpr uint8_t REG_ACCEL_CONFIG = 0x1C;
constexpr uint8_t REG_ACCEL_XOUT_H = 0x3B;
constexpr uint8_t REG_PWR_MGMT_1 = 0x6B;
constexpr uint8_t REG_WHO_AM_I = 0x75;

// MPU6050 Settings
constexpr uint8_t DLPF_CONFIG_42HZ = 0x03;
constexpr uint8_t ACCEL_RANGE_2G = 0x00;
constexpr uint8_t GYRO_RANGE_250DPS = 0x00;
constexpr uint8_t SMPLRT_DIV_100HZ = 9;  // 1kHz / (1 + 9) = 100Hz

struct RawSample {
    uint32_t timestamp_us;
    int16_t ax;
    int16_t ay;
    int16_t az;
    int16_t gx;
    int16_t gy;
    int16_t gz;
};

// Driver functions
void setupI2C();
bool writeRegister(uint8_t addr, uint8_t reg, uint8_t value);
bool i2cReadBlock(uint8_t addr, uint8_t startReg, uint8_t* buf, size_t len);
bool setupMPU6050();
bool readMPU6050Raw(RawSample& outSample);

#endif // MPU6050_DRIVER_H
