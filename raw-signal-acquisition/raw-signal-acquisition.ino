#include <Wire.h>

// =========================
// MPU6050
// =========================
#define MPU6050_ADDR 0x68

// MPU6050 registers
#define PWR_MGMT_1    0x6B
#define SMPLRT_DIV    0x19
#define CONFIG        0x1A
#define GYRO_CONFIG   0x1B
#define ACCEL_CONFIG  0x1C
#define ACCEL_XOUT_H  0x3B

// =========================
// ESP32 I2C pins
// =========================
#define SDA_PIN 21
#define SCL_PIN 22

// Target sampling frequency
#define SAMPLE_RATE_HZ 100
#define SAMPLE_PERIOD_US (1000000UL / SAMPLE_RATE_HZ)

unsigned long nextSampleTime = 0;


// --------------------------------------------------
// Write one byte to MPU6050 register
// --------------------------------------------------
void writeRegister(uint8_t reg, uint8_t value)
{
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(reg);
    Wire.write(value);
    Wire.endTransmission();
}


// --------------------------------------------------
// Read multiple bytes from MPU6050
// --------------------------------------------------
void readMPU6050(uint8_t reg, uint8_t *buffer, uint8_t length)
{
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);

    Wire.requestFrom(MPU6050_ADDR, length);

    for (uint8_t i = 0; i < length; i++)
    {
        if (Wire.available())
        {
            buffer[i] = Wire.read();
        }
    }
}


// --------------------------------------------------
// Initialize MPU6050
// --------------------------------------------------
void setupMPU6050()
{
    // Wake MPU6050
    writeRegister(PWR_MGMT_1, 0x00);

    delay(100);

    /*
       Sample Rate = Gyroscope Output Rate / (1 + SMPLRT_DIV)

       With DLPF enabled:
       Gyroscope Output Rate = 1 kHz

       SMPLRT_DIV = 9

       Therefore:
       1000 / (1 + 9) = 100 Hz
    */
    writeRegister(SMPLRT_DIV, 9);

    /*
       DLPF configuration

       CONFIG = 0x03

       Gyroscope bandwidth = 44 Hz
       Accelerometer bandwidth = 42 Hz

       This provides useful anti-aliasing before sampling.
    */
    writeRegister(CONFIG, 0x03);

    /*
       Gyroscope full scale:
       ±250 °/s

       0x00 = ±250 °/s
    */
    writeRegister(GYRO_CONFIG, 0x00);

    /*
       Accelerometer full scale:
       ±2 g

       0x00 = ±2g
    */
    writeRegister(ACCEL_CONFIG, 0x00);
}


// --------------------------------------------------
// Setup
// --------------------------------------------------
void setup()
{
    Serial.begin(115200);

    Wire.begin(SDA_PIN, SCL_PIN);

    // I2C clock
    Wire.setClock(400000);

    delay(500);

    setupMPU6050();

    // CSV header
    Serial.println("timestamp_us,ax,ay,az,gx,gy,gz");

    nextSampleTime = micros();
}


// --------------------------------------------------
// Main loop
// --------------------------------------------------
void loop()
{
    unsigned long currentTime = micros();

    // Maintain 100 Hz sampling
    if ((long)(currentTime - nextSampleTime) >= 0)
    {
        nextSampleTime += SAMPLE_PERIOD_US;

        uint8_t data[14];

        /*
           MPU6050 data registers:

           0x3B - ACCEL_XOUT_H
           0x3C - ACCEL_XOUT_L
           0x3D - ACCEL_YOUT_H
           0x3E - ACCEL_YOUT_L
           0x3F - ACCEL_ZOUT_H
           0x40 - ACCEL_ZOUT_L
           0x41 - TEMP_OUT_H
           0x42 - TEMP_OUT_L
           0x43 - GYRO_XOUT_H
           0x44 - GYRO_XOUT_L
           0x45 - GYRO_YOUT_H
           0x46 - GYRO_YOUT_L
           0x47 - GYRO_ZOUT_H
           0x48 - GYRO_ZOUT_L
        */

        readMPU6050(ACCEL_XOUT_H, data, 14);

        // Combine high and low bytes
        int16_t ax = ((int16_t)data[0] << 8) | data[1];
        int16_t ay = ((int16_t)data[2] << 8) | data[3];
        int16_t az = ((int16_t)data[4] << 8) | data[5];

        // Skip temperature
        int16_t gx = ((int16_t)data[8] << 8) | data[9];
        int16_t gy = ((int16_t)data[10] << 8) | data[11];
        int16_t gz = ((int16_t)data[12] << 8) | data[13];

        // Timestamp taken at acquisition
        unsigned long timestamp = micros();

        // Send raw data
        Serial.print(timestamp);
        Serial.print(",");

        Serial.print(ax);
        Serial.print(",");

        Serial.print(ay);
        Serial.print(",");

        Serial.print(az);
        Serial.print(",");

        Serial.print(gx);
        Serial.print(",");

        Serial.print(gy);
        Serial.print(",");

        Serial.println(gz);
    }
}
