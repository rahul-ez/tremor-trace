#include "mpu6050_driver.h"
#include <Wire.h>

void setupI2C() {
    Wire.begin(PIN_SDA, PIN_SCL);
    Wire.setClock(I2C_CLOCK_HZ);
}

bool writeRegister(uint8_t addr, uint8_t reg, uint8_t value) {
    Wire.beginTransmission(addr);
    Wire.write(reg);
    Wire.write(value);
    return Wire.endTransmission() == 0;
}

bool i2cReadBlock(uint8_t addr, uint8_t startReg, uint8_t* buf, size_t len) {
    Wire.beginTransmission(addr);
    Wire.write(startReg);
    if (Wire.endTransmission(false) != 0) {
        return false;
    }
    size_t received = Wire.requestFrom(addr, static_cast<uint8_t>(len));
    if (received != len) {
        return false;
    }
    for (size_t i = 0; i < len; i++) {
        buf[i] = Wire.read();
    }
    return true;
}

bool setupMPU6050() {
    setupI2C();

    // Verify WHO_AM_I
    uint8_t whoAmI = 0;
    if (!i2cReadBlock(MPU6050_ADDR, REG_WHO_AM_I, &whoAmI, 1) || whoAmI != 0x68) {
        return false;
    }

    // Wake device
    if (!writeRegister(MPU6050_ADDR, REG_PWR_MGMT_1, 0x00)) return false;
    delay(10);

    // Set sample rate divider (100 Hz output)
    if (!writeRegister(MPU6050_ADDR, REG_SMPLRT_DIV, SMPLRT_DIV_100HZ)) return false;

    // Set DLPF bandwidth (42 Hz accel / 44 Hz gyro)
    if (!writeRegister(MPU6050_ADDR, REG_CONFIG, DLPF_CONFIG_42HZ)) return false;

    // Set gyro range (±250 deg/s)
    if (!writeRegister(MPU6050_ADDR, REG_GYRO_CONFIG, GYRO_RANGE_250DPS)) return false;

    // Set accel range (±2 g)
    if (!writeRegister(MPU6050_ADDR, REG_ACCEL_CONFIG, ACCEL_RANGE_2G)) return false;

    return true;
}

bool readMPU6050Raw(RawSample& outSample) {
    uint8_t buf[14];
    uint32_t t_read = micros();
    if (!i2cReadBlock(MPU6050_ADDR, REG_ACCEL_XOUT_H, buf, sizeof(buf))) {
        return false;
    }
    outSample.timestamp_us = t_read;
    outSample.ax = static_cast<int16_t>((buf[0] << 8) | buf[1]);
    outSample.ay = static_cast<int16_t>((buf[2] << 8) | buf[3]);
    outSample.az = static_cast<int16_t>((buf[4] << 8) | buf[5]);
    // buf[6], buf[7] is temperature (skipped)
    outSample.gx = static_cast<int16_t>((buf[8] << 8) | buf[9]);
    outSample.gy = static_cast<int16_t>((buf[10] << 8) | buf[11]);
    outSample.gz = static_cast<int16_t>((buf[12] << 8) | buf[13]);
    return true;
}
